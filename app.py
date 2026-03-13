import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys
import unittest

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go  # type: ignore
    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    go = None
    PLOTLY_AVAILABLE = False

try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:
    st = None
    STREAMLIT_AVAILABLE = False

try:
    import yfinance as yf  # type: ignore
    YFINANCE_AVAILABLE = True
except ModuleNotFoundError:
    yf = None
    YFINANCE_AVAILABLE = False

APP_TITLE = "Atlas Terminal"
DB_PATH = "signals.db"

TIMEFRAME_MAP = {
    "1H": {"interval": "60m", "period": "90d"},
    "4H": {"interval": "60m", "period": "180d"},
    "1D": {"interval": "1d", "period": "2y"},
    "1W": {"interval": "1wk", "period": "5y"},
}

BIST_30 = [
    "AKBNK.IS", "ALARK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", "EKGYO.IS",
    "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HEKTS.IS",
    "ISCTR.IS", "KCHOL.IS", "KONTR.IS", "KOZAA.IS", "KOZAL.IS", "ODAS.IS",
    "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS",
    "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS", "TUPRS.IS", "YKBNK.IS",
]

BIST_100_EXTRA = [
    "AEFES.IS", "AGHOL.IS", "AKSA.IS", "ARCLK.IS", "BERA.IS", "CIMSA.IS",
    "DOHOL.IS", "ECILC.IS", "EGEEN.IS", "ENERY.IS", "GENIL.IS", "GESAN.IS",
    "HALKB.IS", "ISMEN.IS", "KARSN.IS", "KTLEV.IS", "MAVI.IS", "MGROS.IS",
    "OTKAR.IS", "QUAGR.IS", "SKBNK.IS", "SMRTG.IS", "SOKM.IS", "TKFEN.IS",
    "TTKOM.IS", "ULKER.IS", "VAKBN.IS", "VESBE.IS", "VESTL.IS", "YEOTK.IS",
]

DEFAULT_SYMBOLS = BIST_30.copy()

PROFILE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Korumacı": {
        "description": "Az sinyal, yüksek doğruluk, sıkı filtre.",
        "strength_min": 75,
        "min_rr": 2.0,
        "require_mtf": True,
        "preferred_tf": "4H",
        "daily_target": "1-2",
        "strict_ob": True,
    },
    "Dengeli": {
        "description": "Dengeli fırsat sayısı, makul risk/getiri.",
        "strength_min": 62,
        "min_rr": 1.8,
        "require_mtf": False,
        "preferred_tf": "1H",
        "daily_target": "3-5",
        "strict_ob": False,
    },
    "Agresif": {
        "description": "Daha fazla fırsat, daha erken setup yakalama.",
        "strength_min": 54,
        "min_rr": 1.5,
        "require_mtf": False,
        "preferred_tf": "1H",
        "daily_target": "10+",
        "strict_ob": False,
    },
    "Tarayıcı": {
        "description": "Çok sayıda fırsat listesi, karar sende.",
        "strength_min": 45,
        "min_rr": 1.2,
        "require_mtf": False,
        "preferred_tf": "1H",
        "daily_target": "10+",
        "strict_ob": False,
    },
}

RISK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Düşük": {"sl_mult": 1.2, "tp_mult": 2.4, "risk_tag": "Düşük risk"},
    "Orta": {"sl_mult": 1.5, "tp_mult": 3.0, "risk_tag": "Orta risk"},
    "Agresif": {"sl_mult": 1.9, "tp_mult": 3.8, "risk_tag": "Agresif risk"},
}

STYLE_TAG = "atlas_terminal_v4"


def cache_data_stub(*args, **kwargs):
    def decorator(func: Callable):
        return func
    return decorator


def get_cache_decorator():
    if STREAMLIT_AVAILABLE:
        return st.cache_data
    return cache_data_stub


cache_data = get_cache_decorator()


@dataclass
class AnalysisResult:
    symbol: str
    timeframe: str
    close: float
    ema200: float
    rsi: float
    macd: float
    macd_signal: float
    atr: Optional[float]
    bos: bool
    choch: bool
    structure_trend: str
    liquidity_sweep: bool
    liquidity_label: str
    fvg_count: int
    latest_fvg: Optional[Dict[str, Any]]
    order_block_bias: str
    order_block_bullish_zone: Optional[Tuple[float, float]]
    order_block_bearish_zone: Optional[Tuple[float, float]]
    market_strength: float
    long_probability: float
    short_probability: float
    verdict: str
    verdict_note: str
    mtf_higher_tf: str
    mtf_ok: bool
    mtf_bias: str
    stop_loss: Optional[float]
    take_profit: Optional[float]
    rr_ratio: Optional[float] = None
    profile_name: str = "Dengeli"
    risk_mode: str = "Orta"
    profile_fit: bool = False


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            symbol TEXT,
            timeframe TEXT,
            close REAL,
            verdict TEXT,
            market_strength REAL,
            long_probability REAL,
            short_probability REAL,
            bos INTEGER,
            choch INTEGER,
            liquidity_sweep INTEGER,
            fvg_count INTEGER,
            order_block_bias TEXT,
            stop_loss REAL,
            take_profit REAL,
            note TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_signal(result: AnalysisResult) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO signals (
            created_at, symbol, timeframe, close, verdict, market_strength,
            long_probability, short_probability, bos, choch, liquidity_sweep,
            fvg_count, order_block_bias, stop_loss, take_profit, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            result.symbol,
            result.timeframe,
            result.close,
            result.verdict,
            result.market_strength,
            result.long_probability,
            result.short_probability,
            int(result.bos),
            int(result.choch),
            int(result.liquidity_sweep),
            result.fvg_count,
            result.order_block_bias,
            result.stop_loss,
            result.take_profit,
            result.verdict_note,
        ),
    )
    conn.commit()
    conn.close()


def load_recent_signals(limit: int = 150) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM signals ORDER BY id DESC LIMIT {int(limit)}", conn
    )
    conn.close()
    return df


def safe_float(value) -> Optional[float]:
    if pd.isna(value):
        return None
    return float(value)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        .dropna()
    )


def verdict_badge(verdict: str) -> str:
    mapping = {
        "GÜÇLÜ AL": "🟢 GÜÇLÜ AL",
        "AL / İZLE": "🟩 AL / İZLE",
        "İZLE": "🟨 İZLE",
        "SAT / İZLE": "🟧 SAT / İZLE",
        "GÜÇLÜ SAT": "🔴 GÜÇLÜ SAT",
    }
    return mapping.get(verdict, verdict)


def verdict_color(verdict: str) -> str:
    mapping = {
        "GÜÇLÜ AL": "#1dd1a1",
        "AL / İZLE": "#4ade80",
        "İZLE": "#fbbf24",
        "SAT / İZLE": "#fb923c",
        "GÜÇLÜ SAT": "#f43f5e",
    }
    return mapping.get(verdict, "#94a3b8")


def strength_color(strength: float) -> str:
    if strength >= 80:
        return "#10b981"
    if strength >= 65:
        return "#22c55e"
    if strength >= 50:
        return "#f59e0b"
    if strength >= 35:
        return "#fb923c"
    return "#f43f5e"


def universe_options() -> Dict[str, List[str]]:
    bist100 = sorted(list(set(BIST_30 + BIST_100_EXTRA)))
    return {
        "BIST 30": BIST_30,
        "BIST 100": bist100,
        "Hazır Liste": DEFAULT_SYMBOLS,
    }


def validate_timeframe(tf: str) -> None:
    if tf not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {tf}")


def format_price(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def extract_tv_symbol(symbol: str) -> str:
    return symbol.replace(".IS", "")


def profile_summary(profile_name: str, risk_mode: str) -> Dict[str, Any]:
    profile = PROFILE_CONFIGS[profile_name]
    risk = RISK_CONFIGS[risk_mode]
    return {
        "name": profile_name,
        "description": profile["description"],
        "strength_min": profile["strength_min"],
        "daily_target": profile["daily_target"],
        "preferred_tf": profile["preferred_tf"],
        "risk_tag": risk["risk_tag"],
        "require_mtf": profile["require_mtf"],
    }


@cache_data(ttl=180, show_spinner=False)
def download_symbol(symbol: str, tf: str) -> pd.DataFrame:
    validate_timeframe(tf)
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()

    cfg = TIMEFRAME_MAP[tf]
    df = yf.download(
        tickers=symbol,
        interval=cfg["interval"],
        period=cfg["period"],
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in df.columns for col in required):
        return pd.DataFrame()

    df = df[required].dropna().copy()
    df.index = pd.to_datetime(df.index)

    if tf == "4H":
        df = resample_ohlcv(df, "4H")

    return df


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast = ema(series, 12)
    slow = ema(series, 26)
    macd_line = fast - slow
    signal_line = ema(macd_line, 9)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA50"] = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)
    out["RSI14"] = rsi(out["Close"], 14)
    out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = macd(out["Close"])
    out["ATR14"] = atr(out, 14)
    out["RANGE_PCT"] = ((out["High"] - out["Low"]) / out["Close"].replace(0, np.nan) * 100).fillna(0)
    return out


def pivot_high(high: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    out = pd.Series(False, index=high.index)
    for i in range(left, len(high) - right):
        window = high.iloc[i - left:i + right + 1]
        out.iloc[i] = high.iloc[i] == window.max() and (window == high.iloc[i]).sum() == 1
    return out


def pivot_low(low: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    out = pd.Series(False, index=low.index)
    for i in range(left, len(low) - right):
        window = low.iloc[i - left:i + right + 1]
        out.iloc[i] = low.iloc[i] == window.min() and (window == low.iloc[i]).sum() == 1
    return out


def detect_bos_choch(df: pd.DataFrame) -> Dict[str, Any]:
    ph = pivot_high(df["High"])
    pl = pivot_low(df["Low"])
    swing_highs = df.loc[ph, "High"]
    swing_lows = df.loc[pl, "Low"]

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"bos": False, "choch": False, "trend": "NEUTRAL", "last_swing_high": None, "last_swing_low": None}

    last_high = float(swing_highs.iloc[-1])
    prev_high = float(swing_highs.iloc[-2])
    last_low = float(swing_lows.iloc[-1])
    prev_low = float(swing_lows.iloc[-2])
    close = float(df["Close"].iloc[-1])

    bullish_structure = last_high > prev_high and last_low > prev_low
    bearish_structure = last_high < prev_high and last_low < prev_low
    bos = close > last_high or close < last_low
    choch = (bullish_structure and close < last_low) or (bearish_structure and close > last_high)
    trend = "UP" if bullish_structure else "DOWN" if bearish_structure else "NEUTRAL"

    return {
        "bos": bos,
        "choch": choch,
        "trend": trend,
        "last_swing_high": last_high,
        "last_swing_low": last_low,
    }


def detect_fvg(df: pd.DataFrame, lookback: int = 120) -> List[Dict[str, Any]]:
    rows = df.tail(lookback)
    idxs = list(rows.index)
    gaps: List[Dict[str, Any]] = []
    for i in range(2, len(rows)):
        a = rows.iloc[i - 2]
        c = rows.iloc[i]
        if c["Low"] > a["High"]:
            gaps.append({"type": "bullish", "start": idxs[i - 2], "end": idxs[i], "low": float(a["High"]), "high": float(c["Low"])})
        elif c["High"] < a["Low"]:
            gaps.append({"type": "bearish", "start": idxs[i - 2], "end": idxs[i], "low": float(c["High"]), "high": float(a["Low"])})
    return gaps


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
    if len(df) < lookback + 2:
        return {"bull_sweep": False, "bear_sweep": False, "active": False, "label": "-"}

    prev_high = float(df["High"].iloc[-lookback - 1:-1].max())
    prev_low = float(df["Low"].iloc[-lookback - 1:-1].min())
    last = df.iloc[-1]

    bull_sweep = bool(last["High"] > prev_high and last["Close"] < prev_high)
    bear_sweep = bool(last["Low"] < prev_low and last["Close"] > prev_low)
    active = bull_sweep or bear_sweep
    label = "🧹 Bear Trap" if bear_sweep else "🧹 Bull Trap" if bull_sweep else "-"

    return {
        "bull_sweep": bull_sweep,
        "bear_sweep": bear_sweep,
        "active": active,
        "label": label,
        "prev_high": prev_high,
        "prev_low": prev_low,
    }


def detect_order_blocks(df: pd.DataFrame, lookback: int = 80) -> Dict[str, Any]:
    data = df.tail(lookback).copy()
    if len(data) < 12:
        return {"bias": "NEUTRAL", "bullish_zone": None, "bearish_zone": None}

    bullish_zone = None
    bearish_zone = None

    for i in range(len(data) - 4, 2, -1):
        row = data.iloc[i]
        next3 = data.iloc[i + 1:i + 4]
        if len(next3) >= 2 and row["Close"] < row["Open"] and (next3["Close"] > next3["Open"]).sum() >= 2:
            bullish_zone = (float(row["Low"]), float(row["High"]))
            break

    for i in range(len(data) - 4, 2, -1):
        row = data.iloc[i]
        next3 = data.iloc[i + 1:i + 4]
        if len(next3) >= 2 and row["Close"] > row["Open"] and (next3["Close"] < next3["Open"]).sum() >= 2:
            bearish_zone = (float(row["Low"]), float(row["High"]))
            break

    close = float(data["Close"].iloc[-1])
    bias = "NEUTRAL"
    if bullish_zone and bullish_zone[0] <= close <= bullish_zone[1]:
        bias = "BULLISH_OB"
    elif bearish_zone and bearish_zone[0] <= close <= bearish_zone[1]:
        bias = "BEARISH_OB"
    elif close > float(data["EMA50"].iloc[-1]):
        bias = "BULLISH"
    elif close < float(data["EMA50"].iloc[-1]):
        bias = "BEARISH"

    return {"bias": bias, "bullish_zone": bullish_zone, "bearish_zone": bearish_zone}


def calc_market_strength(df: pd.DataFrame, smc: Dict[str, Any], ob: Dict[str, Any], sweep: Dict[str, Any]) -> Tuple[float, float, float, str]:
    last = df.iloc[-1]
    score = 50.0
    notes: List[str] = []

    if last["Close"] > last["EMA200"]:
        score += 12
        notes.append("EMA200 üstü")
    else:
        score -= 12
        notes.append("EMA200 altı")

    if last["EMA50"] > last["EMA200"]:
        score += 6
        notes.append("EMA50>EMA200")
    else:
        score -= 6
        notes.append("EMA50<EMA200")

    if 52 <= last["RSI14"] <= 68:
        score += 8
        notes.append("RSI sağlıklı")
    elif last["RSI14"] >= 75:
        score -= 4
        notes.append("RSI aşırı alım")
    elif last["RSI14"] <= 30:
        score += 4
        notes.append("RSI aşırı satım tepki")
    elif last["RSI14"] < 45:
        score -= 6
        notes.append("RSI zayıf")

    if last["MACD"] > last["MACD_SIGNAL"]:
        score += 9
        notes.append("MACD pozitif")
    else:
        score -= 9
        notes.append("MACD negatif")

    if smc["trend"] == "UP":
        score += 10
        notes.append("yapı yukarı")
    elif smc["trend"] == "DOWN":
        score -= 10
        notes.append("yapı aşağı")

    if smc["bos"]:
        score += 8 if last["Close"] > last["EMA200"] else -8
        notes.append("BOS")

    if smc["choch"]:
        if smc["trend"] == "UP":
            score -= 10
            notes.append("CHoCH aşağı risk")
        elif smc["trend"] == "DOWN":
            score += 10
            notes.append("CHoCH yukarı dönüş")

    if "BULLISH" in ob["bias"]:
        score += 8
        notes.append("bullish OB")
    elif "BEARISH" in ob["bias"]:
        score -= 8
        notes.append("bearish OB")

    if sweep["bear_sweep"]:
        score += 12
        notes.append("bear trap")
    if sweep["bull_sweep"]:
        score -= 12
        notes.append("bull trap")

    score = max(0, min(100, score))
    long_prob = round(score, 1)
    short_prob = round(100 - score, 1)

    if score >= 76:
        verdict = "GÜÇLÜ AL"
    elif score >= 60:
        verdict = "AL / İZLE"
    elif score <= 24:
        verdict = "GÜÇLÜ SAT"
    elif score <= 40:
        verdict = "SAT / İZLE"
    else:
        verdict = "İZLE"

    return round(score, 1), long_prob, short_prob, f"{verdict} | {', '.join(notes)}"


def build_risk_levels(df: pd.DataFrame, direction: str, risk_mode: str = "Orta") -> Tuple[Optional[float], Optional[float], Optional[float]]:
    last = df.iloc[-1]
    atr_value = safe_float(last["ATR14"])
    close = float(last["Close"])

    if not atr_value or atr_value <= 0:
        return None, None, None

    risk_cfg = RISK_CONFIGS[risk_mode]
    sl_mult = float(risk_cfg["sl_mult"])
    tp_mult = float(risk_cfg["tp_mult"])

    if direction in ["GÜÇLÜ AL", "AL / İZLE"]:
        sl = close - sl_mult * atr_value
        tp = close + tp_mult * atr_value
        rr = (tp - close) / max(close - sl, 1e-9)
    elif direction in ["GÜÇLÜ SAT", "SAT / İZLE"]:
        sl = close + sl_mult * atr_value
        tp = close - tp_mult * atr_value
        rr = (close - tp) / max(sl - close, 1e-9)
    else:
        sl = close - 1.0 * atr_value
        tp = close + 1.5 * atr_value
        rr = (tp - close) / max(close - sl, 1e-9)

    return round(sl, 4), round(tp, 4), round(float(rr), 2)


def mtf_confirmation(symbol: str, base_tf: str) -> Dict[str, Any]:
    tf_order = ["1H", "4H", "1D", "1W"]
    idx = tf_order.index(base_tf)
    higher_tf = tf_order[min(idx + 1, len(tf_order) - 1)]

    base_raw = download_symbol(symbol, base_tf)
    higher_raw = download_symbol(symbol, higher_tf)
    if base_raw.empty or higher_raw.empty:
        return {"ok": False, "higher_tf": higher_tf, "bias": "N/A"}

    base_df = add_indicators(base_raw)
    higher_df = add_indicators(higher_raw)

    base_up = bool(base_df["Close"].iloc[-1] > base_df["EMA200"].iloc[-1])
    higher_up = bool(higher_df["Close"].iloc[-1] > higher_df["EMA200"].iloc[-1])
    aligned = base_up == higher_up

    if aligned and base_up:
        bias = "UYUMLU YUKARI"
    elif aligned and not base_up:
        bias = "UYUMLU AŞAĞI"
    else:
        bias = "ÇATIŞMALI"

    return {"ok": aligned, "higher_tf": higher_tf, "bias": bias}


def apply_profile_filter(result: AnalysisResult, profile_name: str) -> bool:
    cfg = PROFILE_CONFIGS[profile_name]
    if result.market_strength < float(cfg["strength_min"]):
        return False
    if cfg["require_mtf"] and not result.mtf_ok:
        return False
    if cfg["strict_ob"] and "BULLISH" not in result.order_block_bias and "BEARISH" not in result.order_block_bias:
        return False
    if result.rr_ratio is not None and result.rr_ratio < float(cfg["min_rr"]):
        return False
    return True


def analyze_symbol(
    symbol: str,
    timeframe: str,
    profile_name: str = "Dengeli",
    risk_mode: str = "Orta",
) -> Tuple[pd.DataFrame, Optional[AnalysisResult], Optional[str]]:
    raw = download_symbol(symbol, timeframe)
    if raw.empty:
        if not YFINANCE_AVAILABLE:
            return pd.DataFrame(), None, "Veri alınamadı: yfinance kurulu değil."
        return pd.DataFrame(), None, "Veri alınamadı"

    df = add_indicators(raw)
    if len(df) < 50:
        return pd.DataFrame(), None, "Yetersiz veri"

    smc = detect_bos_choch(df)
    fvg = detect_fvg(df)
    sweep = detect_liquidity_sweep(df)
    ob = detect_order_blocks(df)
    market_strength, long_prob, short_prob, verdict_note = calc_market_strength(df, smc, ob, sweep)
    verdict = verdict_note.split("|")[0].strip()
    stop_loss, take_profit, rr_ratio = build_risk_levels(df, verdict, risk_mode)
    mtf = mtf_confirmation(symbol, timeframe)

    result = AnalysisResult(
        symbol=symbol,
        timeframe=timeframe,
        close=round(float(df["Close"].iloc[-1]), 4),
        ema200=round(float(df["EMA200"].iloc[-1]), 4),
        rsi=round(float(df["RSI14"].iloc[-1]), 2),
        macd=round(float(df["MACD"].iloc[-1]), 4),
        macd_signal=round(float(df["MACD_SIGNAL"].iloc[-1]), 4),
        atr=round(float(df["ATR14"].iloc[-1]), 4) if not pd.isna(df["ATR14"].iloc[-1]) else None,
        bos=bool(smc["bos"]),
        choch=bool(smc["choch"]),
        structure_trend=smc["trend"],
        liquidity_sweep=bool(sweep["active"]),
        liquidity_label=sweep["label"],
        fvg_count=len(fvg),
        latest_fvg=fvg[-1] if fvg else None,
        order_block_bias=ob["bias"],
        order_block_bullish_zone=ob["bullish_zone"],
        order_block_bearish_zone=ob["bearish_zone"],
        market_strength=market_strength,
        long_probability=long_prob,
        short_probability=short_prob,
        verdict=verdict,
        verdict_note=verdict_note,
        mtf_higher_tf=mtf["higher_tf"],
        mtf_ok=bool(mtf["ok"]),
        mtf_bias=mtf["bias"],
        stop_loss=stop_loss,
        take_profit=take_profit,
        rr_ratio=rr_ratio,
        profile_name=profile_name,
        risk_mode=risk_mode,
    )
    result.profile_fit = apply_profile_filter(result, profile_name)
    return df, result, None


def plot_chart(df: pd.DataFrame, result: AnalysisResult):
    if not PLOTLY_AVAILABLE:
        return None

    chart_df = df.tail(250).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="Fiyat",
            increasing_line_color="#22c55e",
            decreasing_line_color="#f43f5e",
        )
    )
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA50"], mode="lines", name="EMA50", line=dict(color="#38bdf8", width=1.4)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA200"], mode="lines", name="EMA200", line=dict(color="#fbbf24", width=1.6)))

    if result.latest_fvg:
        fig.add_hrect(y0=result.latest_fvg["low"], y1=result.latest_fvg["high"], line_width=0, fillcolor="#7c3aed", opacity=0.16, annotation_text=f"FVG {result.latest_fvg['type']}")
    if result.order_block_bullish_zone:
        fig.add_hrect(y0=result.order_block_bullish_zone[0], y1=result.order_block_bullish_zone[1], line_width=0, fillcolor="#10b981", opacity=0.10, annotation_text="Bullish OB")
    if result.order_block_bearish_zone:
        fig.add_hrect(y0=result.order_block_bearish_zone[0], y1=result.order_block_bearish_zone[1], line_width=0, fillcolor="#ef4444", opacity=0.10, annotation_text="Bearish OB")
    if result.stop_loss is not None:
        fig.add_hline(y=result.stop_loss, annotation_text="SL", line=dict(color="#fb7185", dash="dot"))
    if result.take_profit is not None:
        fig.add_hline(y=result.take_profit, annotation_text="TP", line=dict(color="#4ade80", dash="dot"))

    fig.update_layout(
        height=680,
        xaxis_rangeslider_visible=False,
        title=f"{result.symbol} | {result.timeframe} | {verdict_badge(result.verdict)} | RR {result.rr_ratio or '-'}",
        legend_orientation="h",
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        margin=dict(l=12, r=12, t=60, b=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.08)")
    return fig


@cache_data(ttl=300, show_spinner=False)
def scan_symbols(symbols: List[str], timeframe: str, profile_name: str, risk_mode: str) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        try:
            _, result, err = analyze_symbol(symbol, timeframe, profile_name, risk_mode)
            if err or result is None or not result.profile_fit:
                continue
            rows.append({
                "Sembol": symbol,
                "TF": timeframe,
                "Fiyat": result.close,
                "Karar": result.verdict,
                "Güç %": result.market_strength,
                "Long %": result.long_probability,
                "Short %": result.short_probability,
                "BOS": "✓" if result.bos else "",
                "CHoCH": "✓" if result.choch else "",
                "Likidite": result.liquidity_label,
                "FVG": result.fvg_count,
                "OB": result.order_block_bias,
                "MTF": result.mtf_bias,
                "RR": result.rr_ratio,
                "Profil": result.profile_name,
                "Risk": result.risk_mode,
                "TradingView": f"https://www.tradingview.com/chart/?symbol=BIST%3A{extract_tv_symbol(symbol)}",
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["Güç %", "Long %"], ascending=[False, False]).reset_index(drop=True)


def build_time_matrix(symbols: List[str], tfs: List[str], profile_name: str, risk_mode: str) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        row = {"Sembol": symbol}
        for tf in tfs:
            try:
                _, result, err = analyze_symbol(symbol, tf, profile_name, risk_mode)
                row[tf] = "Hata" if err or result is None else f"{result.verdict} ({result.market_strength}%)"
            except Exception:
                row[tf] = "Hata"
        rows.append(row)
    return pd.DataFrame(rows)


def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{ font-family: Inter, system-ui, sans-serif; }}
        .stApp {{
            background: radial-gradient(circle at top left, rgba(56,189,248,0.10), transparent 28%),
                        radial-gradient(circle at top right, rgba(20,184,166,0.10), transparent 22%),
                        linear-gradient(180deg, #020617 0%, #071121 100%);
            color: #e2e8f0;
        }}
        .block-container {{ padding-top: 1rem; padding-bottom: 2rem; max-width: 1550px; }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(2,6,23,0.98));
            border-right: 1px solid rgba(148,163,184,0.12);
        }}
        div[data-testid="stMetric"] {{
            background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(15,23,42,0.72));
            border: 1px solid rgba(148,163,184,0.10);
            padding: 14px 16px;
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(2,6,23,0.28);
        }}
        .hero-card {{
            padding: 22px 24px;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(14,165,233,0.18), rgba(15,23,42,0.78) 45%, rgba(16,185,129,0.14));
            border: 1px solid rgba(125,211,252,0.14);
            margin-bottom: 1rem;
            box-shadow: 0 20px 45px rgba(2,6,23,0.32);
        }}
        .hero-title {{ font-size: 2rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.25rem; }}
        .hero-sub {{ color: #94a3b8; font-size: 0.98rem; margin-bottom: 1rem; }}
        .status-row {{ display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; }}
        .status-pill {{ border-radius: 16px; padding: 12px 14px; background: rgba(15,23,42,0.65); border: 1px solid rgba(148,163,184,0.10); }}
        .status-pill .label {{ color: #94a3b8; font-size: 0.8rem; margin-bottom: 4px; }}
        .status-pill .value {{ color: #f8fafc; font-size: 1rem; font-weight: 700; }}
        .glass-card {{ background: linear-gradient(180deg, rgba(15,23,42,0.84), rgba(15,23,42,0.60)); border: 1px solid rgba(148,163,184,0.10); border-radius: 22px; padding: 18px; box-shadow: 0 12px 32px rgba(2,6,23,0.28); }}
        .signal-box {{ padding: 18px; border-radius: 20px; background: linear-gradient(180deg, rgba(15,23,42,0.94), rgba(15,23,42,0.70)); border: 1px solid rgba(148,163,184,0.10); min-height: 100%; }}
        .signal-head {{ font-size: 0.82rem; color: #94a3b8; margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.08em; }}
        .signal-value {{ font-size: 1.25rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.55rem; }}
        .signal-note {{ color: #cbd5e1; font-size: 0.92rem; line-height: 1.5; }}
        .mini-badge {{ display: inline-block; margin-top: 8px; padding: 6px 10px; border-radius: 999px; font-size: 0.78rem; background: rgba(56,189,248,0.10); color: #7dd3fc; border: 1px solid rgba(56,189,248,0.14); }}
        .pulse-wrap {{ display:grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap:12px; margin: 0.7rem 0 1rem 0; }}
        .pulse-card {{ background: rgba(15,23,42,0.74); border:1px solid rgba(148,163,184,0.1); border-radius:16px; padding:14px; }}
        .scanner-row {{ display:grid; grid-template-columns: 1.1fr .65fr .75fr .7fr .7fr .9fr .8fr .6fr; gap:10px; align-items:center; padding:12px 14px; border-radius:16px; background: rgba(15,23,42,0.64); border:1px solid rgba(148,163,184,0.08); margin-bottom:8px; }}
        .scanner-head {{ color:#94a3b8; font-size:0.76rem; text-transform:uppercase; letter-spacing:0.08em; }}
        .scanner-val {{ color:#f8fafc; font-size:0.95rem; font-weight:700; }}
        .heatbar {{ height: 8px; border-radius: 999px; background: rgba(255,255,255,0.06); overflow:hidden; }}
        .heatfill {{ height:100%; border-radius:999px; }}
        .select-btn button {{ width:100%; border-radius:12px; }}
        button[data-baseweb="tab"] {{ height: 46px; border-radius: 14px; background: rgba(15,23,42,0.55); border: 1px solid rgba(148,163,184,0.10); color: #cbd5e1; padding: 0 16px; }}
        button[data-baseweb="tab"][aria-selected="true"] {{ background: linear-gradient(135deg, rgba(14,165,233,0.16), rgba(16,185,129,0.14)); color: #f8fafc; border: 1px solid rgba(125,211,252,0.18); }}
        .footer-note {{ color: #64748b; font-size: 0.85rem; margin-top: 0.8rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(result: Optional[AnalysisResult], selected_symbol: str, timeframe: str, profile_name: str, risk_mode: str) -> None:
    verdict = verdict_badge(result.verdict) if result else "Hazır"
    verdict_hex = verdict_color(result.verdict) if result else "#38bdf8"
    market_strength = f"%{result.market_strength}" if result else "-"
    mtf = result.mtf_bias if result else "-"
    price = format_price(result.close) if result else "-"
    sweep = result.liquidity_label if result else "-"

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-sub">Scanner-first mimari, kullanıcı profiline göre filtrelenmiş fırsatlar ve premium karar ekranı.</div>
            <div class="status-row">
                <div class="status-pill"><div class="label">Profil</div><div class="value">{profile_name}</div></div>
                <div class="status-pill"><div class="label">Risk Modu</div><div class="value">{risk_mode}</div></div>
                <div class="status-pill"><div class="label">Sembol</div><div class="value">{selected_symbol}</div></div>
                <div class="status-pill"><div class="label">Aktif Karar</div><div class="value" style="color:{verdict_hex};">{verdict}</div></div>
            </div>
            <div class="status-row" style="margin-top:12px;">
                <div class="status-pill"><div class="label">Zaman Dilimi</div><div class="value">{timeframe}</div></div>
                <div class="status-pill"><div class="label">Son Fiyat</div><div class="value">{price}</div></div>
                <div class="status-pill"><div class="label">Market Strength</div><div class="value">{market_strength}</div></div>
                <div class="status-pill"><div class="label">Likidite / MTF</div><div class="value">{sweep} · {mtf}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_profile_panel(profile_name: str, risk_mode: str) -> None:
    summary = profile_summary(profile_name, risk_mode)
    st.markdown(
        f"""
        <div class="glass-card" style="margin-bottom:1rem;">
            <div class="signal-head">Üye Profili</div>
            <div class="signal-value">{summary['name']} · {summary['risk_tag']}</div>
            <div class="signal-note">{summary['description']}</div>
            <div class="status-row" style="margin-top:12px;">
                <div class="status-pill"><div class="label">Min Strength</div><div class="value">%{summary['strength_min']}</div></div>
                <div class="status-pill"><div class="label">Günlük Fırsat</div><div class="value">{summary['daily_target']}</div></div>
                <div class="status-pill"><div class="label">Tercih TF</div><div class="value">{summary['preferred_tf']}</div></div>
                <div class="status-pill"><div class="label">MTF Şartı</div><div class="value">{'Evet' if summary['require_mtf'] else 'Hayır'}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_boxes(result: AnalysisResult) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='signal-box'><div class='signal-head'>Piyasa Yapısı</div><div class='signal-value'>{result.structure_trend}</div><div class='signal-note'>BOS: {'Evet' if result.bos else 'Hayır'} · CHoCH: {'Evet' if result.choch else 'Hayır'} · FVG: {result.fvg_count}</div><div class='mini-badge'>{result.order_block_bias}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='signal-box'><div class='signal-head'>Karar Motoru</div><div class='signal-value' style='color:{verdict_color(result.verdict)};'>{verdict_badge(result.verdict)}</div><div class='signal-note'>Long: %{result.long_probability} · Short: %{result.short_probability}</div><div class='mini-badge'>{result.mtf_bias}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='signal-box'><div class='signal-head'>Risk Alanı</div><div class='signal-value'>SL {format_price(result.stop_loss)} / TP {format_price(result.take_profit)}</div><div class='signal-note'>ATR: {format_price(result.atr)} · RR: {result.rr_ratio if result.rr_ratio is not None else '-'}</div><div class='mini-badge'>{result.profile_name} · {result.risk_mode}</div></div>",
            unsafe_allow_html=True,
        )


def sidebar() -> Tuple[List[str], str, bool, int, int, str, str]:
    st.sidebar.markdown("## ⚙️ Kontrol Merkezi")
    predefined = universe_options()
    mode = st.sidebar.radio("Evren", ["Hazır Liste", "Özel Liste"], index=0)

    if mode == "Hazır Liste":
        selected_universe = st.sidebar.selectbox("Liste seç", list(predefined.keys()), index=0)
        symbols = predefined[selected_universe]
    else:
        raw_text = st.sidebar.text_area("Semboller (.IS ile, virgül ile ayır)", value=",".join(BIST_30[:8]), height=130)
        symbols = [s.strip().upper() for s in raw_text.split(",") if s.strip()]

    st.sidebar.markdown("### Üye Profili")
    profile_name = st.sidebar.selectbox("Tarz", list(PROFILE_CONFIGS.keys()), index=1)
    risk_mode = st.sidebar.selectbox("Risk Seviyesi", list(RISK_CONFIGS.keys()), index=1)
    timeframe = st.sidebar.selectbox("Zaman Dilimi", ["1H", "4H", "1D", "1W"], index=0)
    auto_scan = st.sidebar.checkbox("Otomatik yenile", value=False)
    interval = st.sidebar.selectbox("Tarama periyodu (dk)", [5, 15, 30, 60], index=1)
    matrix_count = st.sidebar.slider("Zaman matrisi sembol sayısı", 4, min(20, max(4, len(symbols))), min(8, len(symbols)))

    summary = profile_summary(profile_name, risk_mode)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="padding:12px 14px;border-radius:16px;background:rgba(15,23,42,0.65);border:1px solid rgba(148,163,184,0.10);">
            <div style="color:#e2e8f0;font-weight:700;margin-bottom:6px;">Aktif Profil</div>
            <div style="color:#94a3b8;font-size:0.88rem;line-height:1.5;">{summary['description']}<br>Min strength: %{summary['strength_min']}<br>Günlük fırsat: {summary['daily_target']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return symbols, timeframe, auto_scan, interval, matrix_count, profile_name, risk_mode


def render_summary_cards(result: AnalysisResult) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI Kararı", verdict_badge(result.verdict))
    c2.metric("Market Strength", f"%{result.market_strength}")
    c3.metric("Long Olasılığı", f"%{result.long_probability}")
    c4.metric("Short Olasılığı", f"%{result.short_probability}")

    a, b, c, d = st.columns(4)
    a.metric("RSI", result.rsi)
    b.metric("EMA200", result.ema200)
    c.metric("ATR", result.atr or 0)
    d.metric("RR", result.rr_ratio or 0)


def render_structure_panel(result: AnalysisResult) -> None:
    col_left, col_right = st.columns([1.25, 1])
    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Piyasa Yapısı")
        st.write(f"**Trend:** {result.structure_trend}")
        st.write(f"**BOS:** {'Evet' if result.bos else 'Hayır'}")
        st.write(f"**CHoCH:** {'Evet' if result.choch else 'Hayır'}")
        st.write(f"**Likidite Süpürme:** {result.liquidity_label}")
        st.write(f"**FVG Sayısı:** {result.fvg_count}")
        st.write(f"**Order Block Bias:** {result.order_block_bias}")
        st.write(f"**AI Notu:** {result.verdict_note}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Risk Yönetimi")
        st.write(f"**Fiyat:** {format_price(result.close)}")
        st.write(f"**Stop Loss:** {format_price(result.stop_loss)}")
        st.write(f"**Take Profit:** {format_price(result.take_profit)}")
        st.write(f"**Üst TF:** {result.mtf_higher_tf}")
        st.write(f"**Profil Uyum:** {'Evet' if result.profile_fit else 'Hayır'}")
        st.markdown(f"[TradingView'de Aç](https://www.tradingview.com/chart/?symbol=BIST%3A{extract_tv_symbol(result.symbol)})")
        st.markdown('</div>', unsafe_allow_html=True)


def render_scan_overview(scan_df: pd.DataFrame) -> None:
    strong_buys = int((scan_df["Karar"] == "GÜÇLÜ AL").sum())
    strong_sells = int((scan_df["Karar"] == "GÜÇLÜ SAT").sum())
    avg_strength = round(float(scan_df["Güç %"].mean()), 1)
    avg_rr = round(float(scan_df["RR"].dropna().mean()), 2) if scan_df["RR"].dropna().shape[0] else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tarama Adedi", len(scan_df))
    c2.metric("Güçlü Al", strong_buys)
    c3.metric("Güçlü Sat", strong_sells)
    c4.metric("Ort. RR", avg_rr)
    st.caption(f"Ortalama güç: %{avg_strength}")


def render_market_pulse(scan_df: pd.DataFrame) -> None:
    if scan_df.empty:
        return
    top = scan_df.head(4)
    cards = []
    for _, row in top.iterrows():
        clr = strength_color(float(row["Güç %"]))
        cards.append(
            f"<div class='pulse-card'><div class='signal-head'>Top Mover</div><div class='signal-value'>{row['Sembol']}</div><div class='signal-note'>Karar: <span style='color:{clr};font-weight:700'>{row['Karar']}</span><br>Strength: %{row['Güç %']} · RR: {row['RR']}</div></div>"
        )
    st.markdown(f"<div class='pulse-wrap'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_scanner_table(scan_df: pd.DataFrame, key_prefix: str = "main") -> Optional[str]:
    c1, c2, c3, c4 = st.columns([1.1, 1.0, 1.0, 0.9])
    with c1:
        search_term = st.text_input("Sembol ara", value="", placeholder="Örn: THYAO", key=f"{key_prefix}_search_term")
    with c2:
        verdict_filter = st.selectbox("Karar filtresi", ["Tümü", "GÜÇLÜ AL", "AL / İZLE", "İZLE", "SAT / İZLE", "GÜÇLÜ SAT"], index=0, key=f"{key_prefix}_verdict_filter")
    with c3:
        min_strength = st.slider("Min güç %", 0, 100, 55, key=f"{key_prefix}_min_strength")
    with c4:
        quick_mode = st.selectbox("Quick filtre", ["Tümü", "Sadece Güçlüler", "Long Ağırlıklı", "Short Ağırlıklı"], index=0, key=f"{key_prefix}_quick_mode")

    filtered = scan_df.copy()
    if search_term.strip():
        filtered = filtered[filtered["Sembol"].str.contains(search_term.strip().upper(), na=False)]
    if verdict_filter != "Tümü":
        filtered = filtered[filtered["Karar"] == verdict_filter]
    filtered = filtered[filtered["Güç %"] >= min_strength]
    if quick_mode == "Sadece Güçlüler":
        filtered = filtered[filtered["Karar"].isin(["GÜÇLÜ AL", "GÜÇLÜ SAT"])]
    elif quick_mode == "Long Ağırlıklı":
        filtered = filtered[filtered["Long %"] >= filtered["Short %"]]
    elif quick_mode == "Short Ağırlıklı":
        filtered = filtered[filtered["Short %"] > filtered["Long %"]]

    if filtered.empty:
        st.info("Filtreye uygun sonuç yok.")
        return None

    header = """
    <div class='scanner-row' style='background:rgba(56,189,248,0.08);'>
      <div class='scanner-head'>Sembol</div>
      <div class='scanner-head'>Fiyat</div>
      <div class='scanner-head'>Karar</div>
      <div class='scanner-head'>Strength</div>
      <div class='scanner-head'>RR</div>
      <div class='scanner-head'>Likidite</div>
      <div class='scanner-head'>OB / MTF</div>
      <div class='scanner-head'>Seç</div>
    </div>
    """
    st.markdown(header, unsafe_allow_html=True)

    selected_symbol: Optional[str] = None
    for i, (_, row) in enumerate(filtered.head(25).iterrows()):
        heat = max(0, min(100, float(row["Güç %"])))
        clr = strength_color(float(row["Güç %"]))
        rr_text = "-" if pd.isna(row["RR"]) else str(row["RR"])
        html = f"""
        <div class='scanner-row'>
          <div class='scanner-val'>{row['Sembol']}</div>
          <div class='scanner-val'>{format_price(float(row['Fiyat']))}</div>
          <div class='scanner-val' style='color:{clr};'>{row['Karar']}</div>
          <div>
             <div class='scanner-val'>%{row['Güç %']}</div>
             <div class='heatbar'><div class='heatfill' style='width:{heat}%; background:{clr};'></div></div>
          </div>
          <div class='scanner-val'>{rr_text}</div>
          <div class='scanner-val'>{row['Likidite']}</div>
          <div class='scanner-val'>{row['OB']} · {row['MTF']}</div>
        </div>
        """
        cols = st.columns([10, 1])
        with cols[0]:
            st.markdown(html, unsafe_allow_html=True)
        with cols[1]:
            if st.button("Aç", key=f"{key_prefix}_pick_{row['Sembol']}_{i}", use_container_width=True):
                selected_symbol = str(row["Sembol"])

    top_hits = filtered.head(3)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(top_hits.iterrows()):
        with cols[idx]:
            clr = strength_color(float(row["Güç %"]))
            st.markdown(
                f"<div class='signal-box'><div class='signal-head'>Öne Çıkan Setup</div><div class='signal-value'>{row['Sembol']}</div><div class='signal-note'>Karar: <span style='color:{clr};font-weight:700;'>{row['Karar']}</span><br>Güç: %{row['Güç %']} · RR: {row['RR']}<br>Likidite: {row['Likidite']}</div><div class='mini-badge'>TF {row['TF']} · {row['Profil']}</div></div>",
                unsafe_allow_html=True,
            )

    return selected_symbol


def render_history() -> None:
    st.subheader("Geçmiş Sinyaller (SQLite)")
    hist = load_recent_signals(150)
    if hist.empty:
        st.info("Henüz kayıt yok.")
        return
    st.dataframe(hist, use_container_width=True, hide_index=True)
    st.download_button(
        label="Geçmişi CSV indir",
        data=hist.to_csv(index=False).encode("utf-8-sig"),
        file_name="signals_history.csv",
        mime="text/csv",
        use_container_width=False,
    )


def try_autorefresh(enabled: bool, interval_minutes: int) -> None:
    if not enabled:
        return
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=interval_minutes * 60 * 1000, key="scanner_refresh")
    except Exception:
        st.info("Otomatik yenileme için streamlit-autorefresh kurman gerekiyor.")


def install_block() -> None:
    with st.expander("Kurulum"):
        st.code("pip install streamlit yfinance pandas numpy plotly streamlit-autorefresh openpyxl\nstreamlit run app.py", language="bash")


def notes_block() -> None:
    with st.expander("Notlar / Yol Haritası"):
        st.warning("Bu sürüm gelişmiş MVP'dir. SMC tespitleri pratik kullanım için sadeleştirilmiştir.")
        st.markdown("- Alarm sistemi\n- Telegram / e-posta bildirimleri\n- Sinyal başarı istatistiği\n- EXE ve mobil sürüm")


def main_streamlit() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    inject_custom_css()
    init_db()
    symbols, timeframe, auto_scan, interval, matrix_count, profile_name, risk_mode = sidebar()

    if not YFINANCE_AVAILABLE:
        st.error("Bu uygulama veri çekebilmek için yfinance gerektirir.")
        install_block()
        return

    if "selected_symbol" not in st.session_state:
        st.session_state["selected_symbol"] = symbols[0]
    if st.session_state["selected_symbol"] not in symbols:
        st.session_state["selected_symbol"] = symbols[0]
    if "symbol_select_widget" not in st.session_state:
        st.session_state["symbol_select_widget"] = st.session_state["selected_symbol"]
    if st.session_state["symbol_select_widget"] not in symbols:
        st.session_state["symbol_select_widget"] = st.session_state["selected_symbol"]
    if "pending_symbol_pick" in st.session_state and st.session_state["pending_symbol_pick"] in symbols:
        st.session_state["selected_symbol"] = st.session_state["pending_symbol_pick"]
        st.session_state["symbol_select_widget"] = st.session_state["pending_symbol_pick"]
        del st.session_state["pending_symbol_pick"]

    try_autorefresh(auto_scan, interval)

    left, right = st.columns([2.3, 1])
    with left:
        selected_symbol = st.selectbox("Hisse seç", symbols, index=symbols.index(st.session_state["symbol_select_widget"]), key="symbol_select_widget")
        st.session_state["selected_symbol"] = selected_symbol
    with right:
        analyze_btn = st.button("Analizi Çalıştır", use_container_width=True)

    result_for_hero: Optional[AnalysisResult] = st.session_state.get("last_result")
    render_hero(result_for_hero, selected_symbol, timeframe, profile_name, risk_mode)
    render_profile_panel(profile_name, risk_mode)

    with st.spinner("Scanner hazırlanıyor..."):
        live_scan_df = scan_symbols(symbols, timeframe, profile_name, risk_mode)

    st.markdown("### Atlas Scanner")
    if live_scan_df.empty:
        st.info("Tarama sonucu bulunamadı.")
    else:
        render_market_pulse(live_scan_df)
        render_scan_overview(live_scan_df)
        picked = render_scanner_table(live_scan_df, key_prefix="top")
        if picked:
            st.session_state["pending_symbol_pick"] = picked
            st.success(f"{picked} grafik analiz için seçildi. Grafik Analiz sekmesine geçebilirsin.")
            st.rerun()

    tabs = st.tabs(["🛰️ Scanner", "📊 Grafik Analiz", "🧭 Zaman Matrisi", "🗃️ Geçmiş"])

    with tabs[0]:
        st.subheader("Canlı Fırsat Masası")
        if live_scan_df.empty:
            st.info("Tarama sonucu bulunamadı.")
        else:
            render_market_pulse(live_scan_df)
            render_scan_overview(live_scan_df)
            picked_tab = render_scanner_table(live_scan_df, key_prefix="tab")
            if picked_tab:
                st.session_state["pending_symbol_pick"] = picked_tab
                st.success(f"{picked_tab} grafik analiz için seçildi.")
                st.rerun()

    with tabs[1]:
        active_symbol = st.session_state.get("selected_symbol", selected_symbol)
        if analyze_btn or "ran_once" not in st.session_state or active_symbol != st.session_state.get("last_analyzed_symbol"):
            st.session_state["ran_once"] = True
            df, result, err = analyze_symbol(active_symbol, timeframe, profile_name, risk_mode)
            if err or result is None:
                st.error(err or "Analiz hatası")
            else:
                st.session_state["last_result"] = result
                st.session_state["last_analyzed_symbol"] = active_symbol
                save_signal(result)
                render_hero(result, active_symbol, timeframe, profile_name, risk_mode)
                render_summary_cards(result)
                render_signal_boxes(result)
                fig = plot_chart(df, result)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Grafik için plotly kurulmalı.")
                render_structure_panel(result)
                st.markdown('<div class="footer-note">Bu analiz yatırım tavsiyesi değildir.</div>', unsafe_allow_html=True)
        elif result_for_hero is not None:
            st.info("Son analiz görüntüleniyor. Farklı sembol için scannerdan seçebilir ya da butona basabilirsin.")

    with tabs[2]:
        st.subheader("Zaman Matrisi")
        with st.spinner("Zaman matrisi hazırlanıyor..."):
            matrix_df = build_time_matrix(symbols[:matrix_count], ["1H", "4H", "1D", "1W"], profile_name, risk_mode)
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    with tabs[3]:
        render_history()

    install_block()
    notes_block()


def print_environment_help() -> None:
    print(APP_TITLE)
    print("Bu dosya Streamlit uygulaması olarak çalışır.")
    if not STREAMLIT_AVAILABLE:
        print("- Eksik paket: streamlit")
    if not YFINANCE_AVAILABLE:
        print("- Eksik paket: yfinance")
    if not PLOTLY_AVAILABLE:
        print("- Eksik paket: plotly")


class TestSmcApp(unittest.TestCase):
    def setUp(self) -> None:
        idx = pd.date_range("2025-01-01", periods=80, freq="h")
        base = np.linspace(100, 140, len(idx))
        self.df = pd.DataFrame({
            "Open": base - 0.5,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base,
            "Volume": np.full(len(idx), 1000),
        }, index=idx)
        self.df_ind = add_indicators(self.df)

    def test_safe_float(self):
        self.assertEqual(safe_float(5), 5.0)
        self.assertIsNone(safe_float(np.nan))

    def test_add_indicators_columns(self):
        for col in ["EMA50", "EMA200", "RSI14", "MACD", "MACD_SIGNAL", "ATR14", "RANGE_PCT"]:
            self.assertIn(col, self.df_ind.columns)

    def test_build_risk_levels_returns_values(self):
        sl, tp, rr = build_risk_levels(self.df_ind, "GÜÇLÜ AL", "Orta")
        self.assertIsNotNone(sl)
        self.assertIsNotNone(tp)
        self.assertIsNotNone(rr)
        self.assertLess(sl, float(self.df_ind["Close"].iloc[-1]))
        self.assertGreater(tp, float(self.df_ind["Close"].iloc[-1]))

    def test_validate_timeframe(self):
        validate_timeframe("1H")
        with self.assertRaises(ValueError):
            validate_timeframe("2H")

    def test_verdict_color_known(self):
        self.assertEqual(verdict_color("GÜÇLÜ AL"), "#1dd1a1")

    def test_strength_color(self):
        self.assertEqual(strength_color(82), "#10b981")
        self.assertEqual(strength_color(20), "#f43f5e")

    def test_format_price_none(self):
        self.assertEqual(format_price(None), "-")

    def test_extract_tv_symbol(self):
        self.assertEqual(extract_tv_symbol("THYAO.IS"), "THYAO")

    def test_profile_summary(self):
        s = profile_summary("Dengeli", "Orta")
        self.assertEqual(s["name"], "Dengeli")
        self.assertIn("strength_min", s)

    def test_apply_profile_filter(self):
        result = AnalysisResult(
            symbol="THYAO.IS",
            timeframe="1H",
            close=100,
            ema200=95,
            rsi=60,
            macd=1,
            macd_signal=0.5,
            atr=2,
            bos=True,
            choch=False,
            structure_trend="UP",
            liquidity_sweep=False,
            liquidity_label="-",
            fvg_count=1,
            latest_fvg=None,
            order_block_bias="BULLISH_OB",
            order_block_bullish_zone=(95, 98),
            order_block_bearish_zone=None,
            market_strength=80,
            long_probability=80,
            short_probability=20,
            verdict="GÜÇLÜ AL",
            verdict_note="x",
            mtf_higher_tf="4H",
            mtf_ok=True,
            mtf_bias="UYUMLU YUKARI",
            stop_loss=97,
            take_profit=106,
            rr_ratio=2.0,
            profile_name="Korumacı",
            risk_mode="Orta",
            profile_fit=False,
        )
        self.assertTrue(apply_profile_filter(result, "Korumacı"))


def run_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSmcApp)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    if "--run-tests" in sys.argv:
        return run_tests()
    if STREAMLIT_AVAILABLE:
        main_streamlit()
        return 0
    print_environment_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
