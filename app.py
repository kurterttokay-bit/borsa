import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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


# ============================================================
# APP CONFIG
# ============================================================
APP_TITLE = "SMC Terminal Pro"
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


# ============================================================
# COMPAT / FALLBACKS
# ============================================================
def cache_data_stub(*args, **kwargs):
    def decorator(func: Callable):
        return func
    return decorator


def get_cache_decorator():
    if STREAMLIT_AVAILABLE:
        return st.cache_data
    return cache_data_stub


cache_data = get_cache_decorator()


# ============================================================
# DATA STRUCTURES
# ============================================================
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


# ============================================================
# DATABASE
# ============================================================
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


# ============================================================
# HELPERS
# ============================================================
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


# ============================================================
# DATA LAYER
# ============================================================
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
    missing = [col for col in required if col not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[required].dropna().copy()
    df.index = pd.to_datetime(df.index)

    if tf == "4H":
        df = resample_ohlcv(df, "4H")

    return df


# ============================================================
# INDICATORS
# ============================================================
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


# ============================================================
# SMC DETECTION
# ============================================================
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
        return {
            "bos": False,
            "choch": False,
            "trend": "NEUTRAL",
            "last_swing_high": None,
            "last_swing_low": None,
        }

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
            gaps.append({
                "type": "bullish",
                "start": idxs[i - 2],
                "end": idxs[i],
                "low": float(a["High"]),
                "high": float(c["Low"]),
            })
        elif c["High"] < a["Low"]:
            gaps.append({
                "type": "bearish",
                "start": idxs[i - 2],
                "end": idxs[i],
                "low": float(c["High"]),
                "high": float(a["Low"]),
            })
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

    return {
        "bias": bias,
        "bullish_zone": bullish_zone,
        "bearish_zone": bearish_zone,
    }


# ============================================================
# AI / DECISION ENGINE
# ============================================================
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
        else:
            notes.append("CHoCH")

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


def build_risk_levels(df: pd.DataFrame, direction: str) -> Tuple[Optional[float], Optional[float]]:
    last = df.iloc[-1]
    atr_value = safe_float(last["ATR14"])
    close = float(last["Close"])

    if not atr_value or atr_value <= 0:
        return None, None

    if direction in ["GÜÇLÜ AL", "AL / İZLE"]:
        sl = close - 1.5 * atr_value
        tp = close + 3.0 * atr_value
    elif direction in ["GÜÇLÜ SAT", "SAT / İZLE"]:
        sl = close + 1.5 * atr_value
        tp = close - 3.0 * atr_value
    else:
        sl = close - 1.0 * atr_value
        tp = close + 1.5 * atr_value

    return round(sl, 4), round(tp, 4)


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


def analyze_symbol(symbol: str, timeframe: str) -> Tuple[pd.DataFrame, Optional[AnalysisResult], Optional[str]]:
    raw = download_symbol(symbol, timeframe)
    if raw.empty:
        if not YFINANCE_AVAILABLE:
            return pd.DataFrame(), None, "Veri alınamadı: yfinance kurulu değil. `pip install yfinance` gerekli."
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
    stop_loss, take_profit = build_risk_levels(df, verdict)
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
    )
    return df, result, None


# ============================================================
# CHARTING
# ============================================================
def plot_chart(df: pd.DataFrame, result: AnalysisResult) -> go.Figure:
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
        )
    )
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA50"], mode="lines", name="EMA50"))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA200"], mode="lines", name="EMA200"))

    if result.latest_fvg:
        fig.add_hrect(
            y0=result.latest_fvg["low"],
            y1=result.latest_fvg["high"],
            line_width=0,
            opacity=0.14,
            annotation_text=f"FVG {result.latest_fvg['type']}",
        )

    if result.order_block_bullish_zone:
        fig.add_hrect(
            y0=result.order_block_bullish_zone[0],
            y1=result.order_block_bullish_zone[1],
            line_width=0,
            opacity=0.10,
            annotation_text="Bullish OB",
        )

    if result.order_block_bearish_zone:
        fig.add_hrect(
            y0=result.order_block_bearish_zone[0],
            y1=result.order_block_bearish_zone[1],
            line_width=0,
            opacity=0.10,
            annotation_text="Bearish OB",
        )

    if result.stop_loss is not None:
        fig.add_hline(y=result.stop_loss, annotation_text="SL")
    if result.take_profit is not None:
        fig.add_hline(y=result.take_profit, annotation_text="TP")

    fig.update_layout(
        height=680,
        xaxis_rangeslider_visible=False,
        title=f"{result.symbol} | {result.timeframe} | {verdict_badge(result.verdict)}",
        legend_orientation="h",
    )
    return fig


# ============================================================
# SCANNER
# ============================================================
@cache_data(ttl=300, show_spinner=False)
def scan_symbols(symbols: List[str], timeframe: str) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        try:
            _, result, err = analyze_symbol(symbol, timeframe)
            if err or result is None:
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
                "TradingView": f"https://www.tradingview.com/chart/?symbol=BIST%3A{symbol.replace('.IS', '')}",
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    scan_df = pd.DataFrame(rows)
    return scan_df.sort_values(["Güç %", "Long %"], ascending=[False, False]).reset_index(drop=True)


def build_time_matrix(symbols: List[str], tfs: List[str]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        row = {"Sembol": symbol}
        for tf in tfs:
            try:
                _, result, err = analyze_symbol(symbol, tf)
                row[tf] = "Hata" if err or result is None else f"{result.verdict} ({result.market_strength}%)"
            except Exception:
                row[tf] = "Hata"
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# UI
# ============================================================
def sidebar() -> Tuple[List[str], str, bool, int, int]:
    st.sidebar.title("Kontrol Paneli")

    predefined = universe_options()
    mode = st.sidebar.radio("Evren", ["Hazır Liste", "Özel Liste"], index=0)

    if mode == "Hazır Liste":
        selected_universe = st.sidebar.selectbox("Liste seç", list(predefined.keys()), index=0)
        symbols = predefined[selected_universe]
    else:
        raw_text = st.sidebar.text_area(
            "Semboller (.IS ile, virgül ile ayır)",
            value=",".join(BIST_30[:8]),
            height=130,
        )
        symbols = [s.strip().upper() for s in raw_text.split(",") if s.strip()]

    timeframe = st.sidebar.selectbox("Zaman Dilimi", ["1H", "4H", "1D", "1W"], index=0)
    auto_scan = st.sidebar.checkbox("Otomatik yenile", value=False)
    interval = st.sidebar.selectbox("Tarama periyodu (dk)", [5, 15, 30, 60], index=1)
    matrix_count = st.sidebar.slider("Zaman matrisi sembol sayısı", 4, min(20, max(4, len(symbols))), min(8, len(symbols)))

    st.sidebar.caption("4H verisi 1H veriden yeniden örneklenir. Bazı BIST sembollerinde Yahoo veri gecikmesi/eksikliği olabilir.")
    return symbols, timeframe, auto_scan, interval, matrix_count


def render_summary_cards(result: AnalysisResult) -> None:
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("AI Kararı", verdict_badge(result.verdict))
    top2.metric("Market Strength", f"%{result.market_strength}")
    top3.metric("Long Olasılığı", f"%{result.long_probability}")
    top4.metric("Short Olasılığı", f"%{result.short_probability}")

    a, b, c, d = st.columns(4)
    a.metric("RSI", result.rsi)
    b.metric("EMA200", result.ema200)
    c.metric("ATR", result.atr or 0)
    d.metric("MTF", result.mtf_bias)


def render_structure_panel(result: AnalysisResult) -> None:
    col_left, col_right = st.columns([1.25, 1])

    with col_left:
        st.subheader("Piyasa Yapısı")
        st.write(f"**Trend:** {result.structure_trend}")
        st.write(f"**BOS:** {'Evet' if result.bos else 'Hayır'}")
        st.write(f"**CHoCH:** {'Evet' if result.choch else 'Hayır'}")
        st.write(f"**Likidite Süpürme:** {result.liquidity_label}")
        st.write(f"**FVG Sayısı:** {result.fvg_count}")
        st.write(f"**Order Block Bias:** {result.order_block_bias}")
        st.write(f"**AI Notu:** {result.verdict_note}")

    with col_right:
        st.subheader("Risk Yönetimi")
        st.write(f"**Fiyat:** {result.close}")
        st.write(f"**Stop Loss:** {result.stop_loss}")
        st.write(f"**Take Profit:** {result.take_profit}")
        st.write(f"**Üst TF:** {result.mtf_higher_tf}")
        tv_symbol = result.symbol.replace('.IS', '')
        st.markdown(f"[TradingView'de Aç](https://www.tradingview.com/chart/?symbol=BIST%3A{tv_symbol})")


def render_history() -> None:
    st.subheader("Geçmiş Sinyaller (SQLite)")
    hist = load_recent_signals(150)
    if hist.empty:
        st.info("Henüz kayıt yok.")
        return

    st.dataframe(hist, use_container_width=True)
    st.download_button(
        "Geçmişi CSV indir",
        hist.to_csv(index=False).encode("utf-8-sig"),
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
        st.info("Otomatik yenileme için `pip install streamlit-autorefresh` kurman gerekiyor.")


def install_block() -> None:
    with st.expander("Kurulum"):
        st.code(
            """
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install streamlit yfinance pandas numpy plotly streamlit-autorefresh openpyxl
streamlit run smc_trading_app.py
            """.strip(),
            language="bash",
        )
        st.write("EXE için sonraki aşamada PyInstaller tabanlı launcher ekleyebiliriz.")


def notes_block() -> None:
    with st.expander("Notlar / Yol Haritası"):
        st.warning(
            "Bu sürüm gelişmiş MVP'dir. SMC tespitleri pratik kullanım için sadeleştirilmiştir; kurumsal seviyede tam birebir order-flow motoru değildir."
        )
        st.markdown(
            """
- Sonraki adımda alarm sistemi eklenebilir.
- Telegram / e-posta bildirimleri bağlanabilir.
- Sinyal başarı istatistiği üretilebilir.
- EXE ve sonra mobil uygulama sürümüne geçilebilir.
            """
        )


def main_streamlit() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    init_db()
    symbols, timeframe, auto_scan, interval, matrix_count = sidebar()

    st.title(APP_TITLE)
    st.caption("SMC + klasik teknik analiz + karar destek sistemi + SQLite kayıt altyapısı")

    if not YFINANCE_AVAILABLE:
        st.error("Bu uygulama veri çekebilmek için yfinance gerektirir. Kurulum: `pip install yfinance`")
        install_block()
        return

    try_autorefresh(auto_scan, interval)

    top_left, top_right = st.columns([2, 1])
    with top_left:
        selected_symbol = st.selectbox("Hisse seç", symbols, index=0)
    with top_right:
        analyze_btn = st.button("Analizi Çalıştır", use_container_width=True)

    tabs = st.tabs(["Tekil Analiz", "Tarama", "Zaman Matrisi", "Geçmiş"])

    with tabs[0]:
        if analyze_btn or "ran_once" not in st.session_state:
            st.session_state["ran_once"] = True
            df, result, err = analyze_symbol(selected_symbol, timeframe)
            if err or result is None:
                st.error(err or "Analiz hatası")
            else:
                save_signal(result)
                render_summary_cards(result)
                st.plotly_chart(plot_chart(df, result), use_container_width=True)
                render_structure_panel(result)

    with tabs[1]:
        st.subheader("Tarama Sonuçları")
        with st.spinner("Hisseler taranıyor..."):
            scan_df = scan_symbols(symbols, timeframe)
        if scan_df.empty:
            st.info("Tarama sonucu bulunamadı.")
        else:
            st.dataframe(scan_df, use_container_width=True)

    with tabs[2]:
        st.subheader("Zaman Matrisi")
        with st.spinner("Zaman matrisi hazırlanıyor..."):
            matrix_df = build_time_matrix(symbols[:matrix_count], ["1H", "4H", "1D", "1W"])
        st.dataframe(matrix_df, use_container_width=True)

    with tabs[3]:
        render_history()

    install_block()
    notes_block()


# ============================================================
# CLI / TESTS
# ============================================================
def print_environment_help() -> None:
    print(f"{APP_TITLE}")
    print("Bu dosya normalde Streamlit uygulaması olarak çalışır.")
    if not STREAMLIT_AVAILABLE:
        print("- Eksik paket: streamlit")
    if not YFINANCE_AVAILABLE:
        print("- Eksik paket: yfinance")
    print("Kurulum:")
    print("  pip install streamlit yfinance pandas numpy plotly streamlit-autorefresh openpyxl")
    print("Çalıştırma:")
    print("  streamlit run smc_trading_app.py")
    print("Testler:")
    print("  python smc_trading_app.py --run-tests")


class TestSmcApp(unittest.TestCase):
    def setUp(self) -> None:
        idx = pd.date_range("2025-01-01", periods=80, freq="H")
        base = np.linspace(100, 140, len(idx))
        self.df = pd.DataFrame(
            {
                "Open": base - 0.5,
                "High": base + 1.0,
                "Low": base - 1.0,
                "Close": base,
                "Volume": np.full(len(idx), 1000),
            },
            index=idx,
        )
        self.df_ind = add_indicators(self.df)

    def test_safe_float(self):
        self.assertEqual(safe_float(5), 5.0)
        self.assertIsNone(safe_float(np.nan))

    def test_add_indicators_columns(self):
        for col in ["EMA50", "EMA200", "RSI14", "MACD", "MACD_SIGNAL", "ATR14", "RANGE_PCT"]:
            self.assertIn(col, self.df_ind.columns)

    def test_build_risk_levels_returns_values(self):
        sl, tp = build_risk_levels(self.df_ind, "GÜÇLÜ AL")
        self.assertIsNotNone(sl)
        self.assertIsNotNone(tp)
        self.assertLess(sl, float(self.df_ind["Close"].iloc[-1]))
        self.assertGreater(tp, float(self.df_ind["Close"].iloc[-1]))

    def test_detect_fvg_returns_list(self):
        gaps = detect_fvg(self.df_ind)
        self.assertIsInstance(gaps, list)

    def test_validate_timeframe(self):
        validate_timeframe("1H")
        with self.assertRaises(ValueError):
            validate_timeframe("2H")

    def test_download_symbol_without_yfinance_returns_empty(self):
        global YFINANCE_AVAILABLE
        old = YFINANCE_AVAILABLE
        YFINANCE_AVAILABLE = False
        try:
            df = download_symbol("THYAO.IS", "1H")
            self.assertTrue(df.empty)
        finally:
            YFINANCE_AVAILABLE = old


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
