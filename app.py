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


def verdict_color(verdict: str) -> str:
    mapping = {
        "GÜÇLÜ AL": "#1dd1a1",
        "AL / İZLE": "#4ade80",
        "İZLE": "#fbbf24",
        "SAT / İZLE": "#fb923c",
        "GÜÇLÜ SAT": "#f43f5e",
    }
    return mapping.get(verdict, "#94a3b8")


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
def plot_chart(df: pd.DataFrame, result: AnalysisResult):
    if not PLOTLY_AVAILABLE:
        return None

    chart_df = df.tail(250).copy()
    candle_up = "#22c55e"
    candle_down = "#f43f5e"

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="Fiyat",
            increasing_line_color=candle_up,
            decreasing_line_color=candle_down,
        )
    )
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA50"], mode="lines", name="EMA50", line=dict(color="#38bdf8", width=1.4)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA200"], mode="lines", name="EMA200", line=dict(color="#fbbf24", width=1.6)))

    if result.latest_fvg:
        fig.add_hrect(
            y0=result.latest_fvg["low"],
            y1=result.latest_fvg["high"],
            line_width=0,
            fillcolor="#7c3aed",
            opacity=0.16,
            annotation_text=f"FVG {result.latest_fvg['type']}",
            annotation_font_color="#e2e8f0",
        )

    if result.order_block_bullish_zone:
        fig.add_hrect(
            y0=result.order_block_bullish_zone[0],
            y1=result.order_block_bullish_zone[1],
            line_width=0,
            fillcolor="#10b981",
            opacity=0.10,
            annotation_text="Bullish OB",
            annotation_font_color="#cbd5e1",
        )

    if result.order_block_bearish_zone:
        fig.add_hrect(
            y0=result.order_block_bearish_zone[0],
            y1=result.order_block_bearish_zone[1],
            line_width=0,
            fillcolor="#ef4444",
            opacity=0.10,
            annotation_text="Bearish OB",
            annotation_font_color="#cbd5e1",
        )

    if result.stop_loss is not None:
        fig.add_hline(y=result.stop_loss, annotation_text="SL", line=dict(color="#fb7185", dash="dot"))
    if result.take_profit is not None:
        fig.add_hline(y=result.take_profit, annotation_text="TP", line=dict(color="#4ade80", dash="dot"))

    fig.update_layout(
        height=680,
        xaxis_rangeslider_visible=False,
        title=f"{result.symbol} | {result.timeframe} | {verdict_badge(result.verdict)}",
        legend_orientation="h",
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        margin=dict(l=15, r=15, t=60, b=15),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.08)")
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
                "TradingView": f"https://www.tradingview.com/chart/?symbol=BIST%3A{extract_tv_symbol(symbol)}",
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
# UI THEME
# ============================================================
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56,189,248,0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(20,184,166,0.10), transparent 22%),
                linear-gradient(180deg, #020617 0%, #071121 100%);
            color: #e2e8f0;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(2,6,23,0.98));
            border-right: 1px solid rgba(148,163,184,0.12);
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(15,23,42,0.72));
            border: 1px solid rgba(148,163,184,0.10);
            padding: 14px 16px;
            border-radius: 18px;
            box-shadow: 0 10px 35px rgba(2,6,23,0.28);
        }
        div[data-testid="stMetricLabel"] {
            color: #94a3b8;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc;
        }
        .hero-card {
            padding: 22px 24px;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(14,165,233,0.18), rgba(15,23,42,0.78) 45%, rgba(16,185,129,0.14));
            border: 1px solid rgba(125,211,252,0.14);
            box-shadow: 0 20px 45px rgba(2,6,23,0.30);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.05rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #f8fafc;
            margin-bottom: 0.25rem;
        }
        .hero-sub {
            color: #94a3b8;
            font-size: 0.98rem;
            margin-bottom: 1rem;
        }
        .status-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0,1fr));
            gap: 12px;
        }
        .status-pill {
            border-radius: 16px;
            padding: 12px 14px;
            background: rgba(15,23,42,0.65);
            border: 1px solid rgba(148,163,184,0.10);
        }
        .status-pill .label {
            color: #94a3b8;
            font-size: 0.8rem;
            margin-bottom: 4px;
        }
        .status-pill .value {
            color: #f8fafc;
            font-size: 1rem;
            font-weight: 700;
        }
        .glass-card {
            background: linear-gradient(180deg, rgba(15,23,42,0.84), rgba(15,23,42,0.60));
            border: 1px solid rgba(148,163,184,0.10);
            border-radius: 22px;
            padding: 18px 18px 12px 18px;
            box-shadow: 0 12px 32px rgba(2,6,23,0.28);
        }
        .signal-box {
            padding: 18px;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(15,23,42,0.94), rgba(15,23,42,0.70));
            border: 1px solid rgba(148,163,184,0.10);
            min-height: 100%;
        }
        .signal-head {
            font-size: 0.82rem;
            color: #94a3b8;
            margin-bottom: 0.4rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .signal-value {
            font-size: 1.25rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.55rem;
        }
        .signal-note {
            color: #cbd5e1;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .mini-badge {
            display: inline-block;
            margin-top: 8px;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            background: rgba(56,189,248,0.10);
            color: #7dd3fc;
            border: 1px solid rgba(56,189,248,0.14);
        }
        div[data-baseweb="tab-list"] {
            gap: 10px;
            background: transparent;
        }
        button[data-baseweb="tab"] {
            height: 46px;
            border-radius: 14px;
            background: rgba(15,23,42,0.55);
            border: 1px solid rgba(148,163,184,0.10);
            color: #cbd5e1;
            padding: 0 16px;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(14,165,233,0.16), rgba(16,185,129,0.14));
            color: #f8fafc;
            border: 1px solid rgba(125,211,252,0.18);
        }
        .stDataFrame, .stTable {
            border-radius: 18px;
            overflow: hidden;
        }
        .footer-note {
            color: #64748b;
            font-size: 0.85rem;
            margin-top: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(result: Optional[AnalysisResult], selected_symbol: str, timeframe: str) -> None:
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
            <div class="hero-sub">Kurumsal yapı okuma, teknik filtreleme ve karar destek sistemini tek ekranda birleştiren BIST terminali.</div>
            <div class="status-row">
                <div class="status-pill">
                    <div class="label">Sembol</div>
                    <div class="value">{selected_symbol}</div>
                </div>
                <div class="status-pill">
                    <div class="label">Zaman Dilimi</div>
                    <div class="value">{timeframe}</div>
                </div>
                <div class="status-pill">
                    <div class="label">Son Fiyat</div>
                    <div class="value">{price}</div>
                </div>
                <div class="status-pill">
                    <div class="label">Aktif Karar</div>
                    <div class="value" style="color:{verdict_hex};">{verdict}</div>
                </div>
            </div>
            <div class="status-row" style="margin-top:12px;">
                <div class="status-pill">
                    <div class="label">Market Strength</div>
                    <div class="value">{market_strength}</div>
                </div>
                <div class="status-pill">
                    <div class="label">MTF Durumu</div>
                    <div class="value">{mtf}</div>
                </div>
                <div class="status-pill">
                    <div class="label">Likidite</div>
                    <div class="value">{sweep}</div>
                </div>
                <div class="status-pill">
                    <div class="label">Veri Kaynağı</div>
                    <div class="value">Yahoo Finance</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_boxes(result: AnalysisResult) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="signal-box">
                <div class="signal-head">Piyasa Yapısı</div>
                <div class="signal-value">{result.structure_trend}</div>
                <div class="signal-note">BOS: {'Evet' if result.bos else 'Hayır'} · CHoCH: {'Evet' if result.choch else 'Hayır'} · FVG sayısı: {result.fvg_count}</div>
                <div class="mini-badge">{result.order_block_bias}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="signal-box">
                <div class="signal-head">Karar Motoru</div>
                <div class="signal-value" style="color:{verdict_color(result.verdict)};">{verdict_badge(result.verdict)}</div>
                <div class="signal-note">Long: %{result.long_probability} · Short: %{result.short_probability}</div>
                <div class="mini-badge">{result.mtf_bias}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="signal-box">
                <div class="signal-head">Risk Alanı</div>
                <div class="signal-value">SL {format_price(result.stop_loss)} / TP {format_price(result.take_profit)}</div>
                <div class="signal-note">ATR: {format_price(result.atr)} · Likidite: {result.liquidity_label}</div>
                <div class="mini-badge">Üst TF: {result.mtf_higher_tf}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# UI
# ============================================================
def sidebar() -> Tuple[List[str], str, bool, int, int]:
    st.sidebar.markdown("## ⚙️ Kontrol Merkezi")

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

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="padding:12px 14px;border-radius:16px;background:rgba(15,23,42,0.65);border:1px solid rgba(148,163,184,0.10);">
            <div style="color:#e2e8f0;font-weight:700;margin-bottom:6px;">Canlı Not</div>
            <div style="color:#94a3b8;font-size:0.88rem;line-height:1.5;">4H veri 1H üzerinden yeniden örneklenir. Yahoo kaynaklı bazı BIST sembollerinde gecikme ya da boş veri görülebilir.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
        tv_symbol = extract_tv_symbol(result.symbol)
        st.markdown(f"[TradingView'de Aç](https://www.tradingview.com/chart/?symbol=BIST%3A{tv_symbol})")
        st.markdown('</div>', unsafe_allow_html=True)


def render_scan_overview(scan_df: pd.DataFrame) -> None:
    if scan_df.empty:
        return

    strong_buys = int((scan_df["Karar"] == "GÜÇLÜ AL").sum())
    strong_sells = int((scan_df["Karar"] == "GÜÇLÜ SAT").sum())
    avg_strength = round(float(scan_df["Güç %"].mean()), 1)
    best_symbol = str(scan_df.iloc[0]["Sembol"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tarama Adedi", len(scan_df))
    c2.metric("Güçlü Al", strong_buys)
    c3.metric("Güçlü Sat", strong_sells)
    c4.metric("Ortalama Güç", f"%{avg_strength}")
    st.caption(f"En yüksek skorlu sembol: {best_symbol}")


def render_history() -> None:
    st.subheader("Geçmiş Sinyaller (SQLite)")
    hist = load_recent_signals(150)
    if hist.empty:
        st.info("Henüz kayıt yok.")
        return

    st.dataframe(hist, use_container_width=True, hide_index=True)
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
    inject_custom_css()
    init_db()
    symbols, timeframe, auto_scan, interval, matrix_count = sidebar()

    if not YFINANCE_AVAILABLE:
        st.error("Bu uygulama veri çekebilmek için yfinance gerektirir. Kurulum: `pip install yfinance`")
        install_block()
        return

    try_autorefresh(auto_scan, interval)

    top_left, top_right = st.columns([2.3, 1])
    with top_left:
        selected_symbol = st.selectbox("Hisse seç", symbols, index=0)
    with top_right:
        analyze_btn = st.button("Analizi Çalıştır", use_container_width=True)

    result_for_hero: Optional[AnalysisResult] = st.session_state.get("last_result")
    render_hero(result_for_hero, selected_symbol, timeframe)

    tabs = st.tabs(["📊 Tekil Analiz", "🛰️ Tarama", "🧭 Zaman Matrisi", "🗃️ Geçmiş"])

    with tabs[0]:
        if analyze_btn or "ran_once" not in st.session_state:
            st.session_state["ran_once"] = True
            df, result, err = analyze_symbol(selected_symbol, timeframe)
            if err or result is None:
                st.error(err or "Analiz hatası")
            else:
                st.session_state["last_result"] = result
                save_signal(result)
                render_hero(result, selected_symbol, timeframe)
                render_summary_cards(result)
                render_signal_boxes(result)
                fig = plot_chart(df, result)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Grafik için plotly kurulmalı.")
                render_structure_panel(result)
                st.markdown('<div class="footer-note">Bu analiz, teknik ve yapısal sinyalleri birleştirerek karar desteği üretir; yatırım tavsiyesi değildir.</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.subheader("Tarama Sonuçları")
        with st.spinner("Hisseler taranıyor..."):
            scan_df = scan_symbols(symbols, timeframe)
        if scan_df.empty:
            st.info("Tarama sonucu bulunamadı.")
        else:
            render_scan_overview(scan_df)
            st.dataframe(scan_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Zaman Matrisi")
        with st.spinner("Zaman matrisi hazırlanıyor..."):
            matrix_df = build_time_matrix(symbols[:matrix_count], ["1H", "4H", "1D", "1W"])
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

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
    if not PLOTLY_AVAILABLE:
        print("- Eksik paket: plotly")
    print("Kurulum:")
    print("  pip install streamlit yfinance pandas numpy plotly streamlit-autorefresh openpyxl")
    print("Çalıştırma:")
    print("  streamlit run smc_trading_app.py")
    print("Testler:")
    print("  python smc_trading_app.py --run-tests")


class TestSmcApp(unittest.TestCase):
    def setUp(self) -> None:
        idx = pd.date_range("2025-01-01", periods=80, freq="h")
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

    def test_verdict_color_known(self):
        self.assertEqual(verdict_color("GÜÇLÜ AL"), "#1dd1a1")

    def test_format_price_none(self):
        self.assertEqual(format_price(None), "-")



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
