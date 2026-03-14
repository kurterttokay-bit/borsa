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

APP_TITLE = "Atlas Money"
APP_SUBTITLE = "Piyasaları Sadeleştirir"
DB_PATH = "signals.db"
INITIAL_CASH = 100.0

TIMEFRAME_MAP = {
    "1G": {"interval": "1d", "period": "2y"},
}

US_MEGA_CAPS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AMD", "NFLX", "AVGO",
    "PLTR", "CRM", "ORCL", "ADBE", "INTC", "QCOM", "MU", "SMCI", "PANW", "SNOW",
]
US_GROWTH = [
    "COIN", "SHOP", "UBER", "ABNB", "CRWD", "DDOG", "ZS", "NET", "RBLX", "SOFI",
    "ARM", "MRVL", "MSTR", "PYPL", "SQ", "CFLT", "FSLR", "ENPH", "DKNG", "ROKU",
]
US_ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "SMH"]
SAFE_ASSETS = ["GLD", "IAU", "TLT", "IEF", "SGOV", "BIL", "SLV"]
RISKY_UNIVERSE = US_MEGA_CAPS + US_GROWTH + US_ETFS
SEARCHABLE_ASSETS = sorted(list(set(RISKY_UNIVERSE + SAFE_ASSETS)))
DEFAULT_SYMBOLS = US_MEGA_CAPS.copy()

PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Koruma Odaklı": {"threshold": 70, "risk_label": "Düşük", "max_positions": 5},
    "Dengeli": {"threshold": 62, "risk_label": "Orta", "max_positions": 5},
    "Büyüme Odaklı": {"threshold": 55, "risk_label": "Orta", "max_positions": 5},
}


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
    market_strength: float
    long_probability: float
    short_probability: float
    verdict: str
    mtf_bias: str
    stop_loss: Optional[float]
    take_profit: Optional[float]
    rr_ratio: Optional[float]


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            symbol TEXT,
            entry_price REAL,
            quantity REAL,
            note TEXT,
            source TEXT DEFAULT 'KULLANICI',
            target_weight REAL DEFAULT 0,
            status TEXT DEFAULT 'open'
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            opened_at TEXT,
            closed_at TEXT,
            symbol TEXT,
            entry_price REAL,
            exit_price REAL,
            quantity REAL,
            pnl REAL,
            pnl_pct REAL,
            note TEXT,
            source TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def get_app_state(key: str, default: str = "") -> str:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
    conn.close()
    return default if row is None else str(row[0])


def set_app_state(key: str, value: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO app_state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )
    conn.commit()
    conn.close()


def get_initial_cash() -> float:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT value FROM app_state WHERE key='initial_cash'").fetchone()
    conn.close()
    if row is None:
        return INITIAL_CASH
    try:
        return float(row[0])
    except Exception:
        return INITIAL_CASH


def set_initial_cash(value: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO app_state (key, value) VALUES ('initial_cash', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(float(value)),),
    )
    conn.commit()
    conn.close()


def add_portfolio_position(
    symbol: str,
    entry_price: float,
    quantity: float,
    note: str = "",
    source: str = "KULLANICI",
    target_weight: float = 0.0,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO portfolio (created_at, symbol, entry_price, quantity, note, source, target_weight, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'open')
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            symbol,
            float(entry_price),
            float(quantity),
            note,
            source,
            float(target_weight),
        ),
    )
    conn.commit()
    conn.close()


def load_open_portfolio() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM portfolio WHERE status='open' ORDER BY id DESC", conn)
    conn.close()
    return df


def load_history(limit: int = 100) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM trades_history ORDER BY id DESC LIMIT {int(limit)}", conn)
    conn.close()
    return df


def close_portfolio_position(position_id: int, exit_price: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    row = cur.execute(
        "SELECT created_at, symbol, entry_price, quantity, note, source FROM portfolio WHERE id = ?",
        (int(position_id),),
    ).fetchone()
    if row is None:
        conn.close()
        return
    opened_at, symbol, entry_price, quantity, note, source = row
    pnl = (float(exit_price) - float(entry_price)) * float(quantity)
    pnl_pct = ((float(exit_price) / float(entry_price)) - 1) * 100 if float(entry_price) else 0.0
    cur.execute(
        """
        INSERT INTO trades_history (opened_at, closed_at, symbol, entry_price, exit_price, quantity, pnl, pnl_pct, note, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            opened_at,
            datetime.now().isoformat(timespec="seconds"),
            symbol,
            float(entry_price),
            float(exit_price),
            float(quantity),
            float(pnl),
            float(pnl_pct),
            note,
            source,
        ),
    )
    cur.execute("UPDATE portfolio SET status='closed' WHERE id = ?", (int(position_id),))
    conn.commit()
    conn.close()


def safe_float(value) -> Optional[float]:
    if pd.isna(value):
        return None
    return float(value)


def format_price(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def action_color(action: str) -> str:
    if action == "TUT":
        return "#22C55E"
    if action in ["SATIŞ DÜŞÜN", "AZALT / DİKKAT"]:
        return "#F59E0B"
    if action == "ÇIK":
        return "#EF4444"
    return "#94A3B8"


@cache_data(ttl=180, show_spinner=False)
def download_symbol(symbol: str, tf: str) -> pd.DataFrame:
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
    return df[required].dropna().copy()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def macd(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    fast = ema(series, 12)
    slow = ema(series, 26)
    line = fast - slow
    signal = ema(line, 9)
    return line, signal


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
    out["MACD"], out["MACD_SIGNAL"] = macd(out["Close"])
    out["ATR14"] = atr(out, 14)
    return out


def build_risk_levels(df: pd.DataFrame, verdict: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    last = df.iloc[-1]
    close = float(last["Close"])
    atr_value = safe_float(last["ATR14"])
    if not atr_value or atr_value <= 0:
        return None, None, None
    sl = close - (1.5 * atr_value)
    tp = close + (3.0 * atr_value)
    rr = (tp - close) / max(close - sl, 1e-9)
    if verdict in ["SAT", "GÜÇLÜ SAT"]:
        sl = close + (1.5 * atr_value)
        tp = close - (3.0 * atr_value)
        rr = (close - tp) / max(sl - close, 1e-9)
    return round(sl, 4), round(tp, 4), round(rr, 2)


def analyze_symbol(symbol: str, timeframe: str = "1G") -> Tuple[pd.DataFrame, Optional[AnalysisResult], Optional[str]]:
    raw = download_symbol(symbol, timeframe)
    if raw.empty:
        return pd.DataFrame(), None, "Veri alınamadı"
    df = add_indicators(raw)
    if len(df) < 50:
        return pd.DataFrame(), None, "Yetersiz veri"
    last = df.iloc[-1]
    score = 50.0
    if last["Close"] > last["EMA200"]:
        score += 15
    else:
        score -= 15
    if last["EMA50"] > last["EMA200"]:
        score += 8
    else:
        score -= 8
    if 50 <= last["RSI14"] <= 68:
        score += 8
    elif last["RSI14"] < 40:
        score -= 7
    if last["MACD"] > last["MACD_SIGNAL"]:
        score += 9
    else:
        score -= 9
    score = max(0, min(100, score))
    if score >= 76:
        verdict = "GÜÇLÜ AL"
    elif score >= 60:
        verdict = "AL"
    elif score <= 24:
        verdict = "GÜÇLÜ SAT"
    elif score <= 40:
        verdict = "SAT"
    else:
        verdict = "İZLE"
    sl, tp, rr = build_risk_levels(df, verdict)
    mtf_bias = "UYUMLU YUKARI" if float(last["Close"]) > float(last["EMA200"]) else "UYUMLU AŞAĞI"
    result = AnalysisResult(
        symbol=symbol,
        timeframe=timeframe,
        close=round(float(last["Close"]), 4),
        ema200=round(float(last["EMA200"]), 4),
        rsi=round(float(last["RSI14"]), 2),
        macd=round(float(last["MACD"]), 4),
        macd_signal=round(float(last["MACD_SIGNAL"]), 4),
        atr=round(float(last["ATR14"]), 4) if not pd.isna(last["ATR14"]) else None,
        market_strength=round(score, 1),
        long_probability=round(score, 1),
        short_probability=round(100 - score, 1),
        verdict=verdict,
        mtf_bias=mtf_bias,
        stop_loss=sl,
        take_profit=tp,
        rr_ratio=rr,
    )
    return df, result, None


def build_radar(profile_name: str, timeframe: str = "1G", limit: int = 3) -> pd.DataFrame:
    threshold = PROFILE_PRESETS[profile_name]["threshold"]
    rows = []
    for symbol in RISKY_UNIVERSE:
        try:
            _, result, err = analyze_symbol(symbol, timeframe)
            if err or result is None:
                continue
            if result.market_strength < threshold:
                continue
            if result.verdict not in ["GÜÇLÜ AL", "AL"]:
                continue
            rows.append(
                {
                    "Sembol": symbol,
                    "Atlas Skoru": result.market_strength,
                    "Karar": result.verdict,
                    "Fiyat": result.close,
                    "Risk/Getiri": result.rr_ratio,
                    "Stop": result.stop_loss,
                    "Hedef": result.take_profit,
                }
            )
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Atlas Skoru", "Risk/Getiri"], ascending=[False, False]).head(limit).reset_index(drop=True)


def suggest_initial_portfolio(profile_name: str) -> List[Dict[str, Any]]:
    radar = build_radar(profile_name, "1G", 6)
    stock_rows: List[Dict[str, Any]] = []
    if not radar.empty:
        for _, row in radar.head(3).iterrows():
            stock_rows.append(
                {
                    "Varlık": str(row["Sembol"]),
                    "Tür": "Hisse",
                    "Pay": 0.20,
                    "Pay Yazı": "%20",
                    "Not": f"Atlas Skoru %{row['Atlas Skoru']} · Risk/Getiri {row['Risk/Getiri']}",
                }
            )
    while len(stock_rows) < 3:
        fallback = DEFAULT_SYMBOLS[len(stock_rows)]
        stock_rows.append(
            {
                "Varlık": fallback,
                "Tür": "Hisse",
                "Pay": 0.20,
                "Pay Yazı": "%20",
                "Not": "Başlangıç portföyü için güçlü ve likit aday",
            }
        )
    return stock_rows + [
        {"Varlık": "GLD", "Tür": "Altın ETF", "Pay": 0.20, "Pay Yazı": "%20", "Not": "Portföy dengesi ve güvenli liman"},
        {"Varlık": "SGOV", "Tür": "PPF / Nakit Park", "Pay": 0.20, "Pay Yazı": "%20", "Not": "Boşta bekleyen fırsat sermayesi"},
    ]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] { font-family: Inter, system-ui, sans-serif; }
        .stApp { background: linear-gradient(180deg, #0B1220 0%, #111827 100%); color:#E5E7EB; }
        .block-container { max-width: 1320px; padding-top: 1rem; padding-bottom: 2rem; }
        section[data-testid="stSidebar"] { background: #0B1220; border-right: 1px solid rgba(148,163,184,0.12); }
        .topbar { display:flex; justify-content:space-between; align-items:center; padding:16px 18px; border-radius:20px; background:#111827; border:1px solid rgba(148,163,184,0.10); margin-bottom:16px; }
        .brand-title { font-size:1.45rem; font-weight:800; color:#F8FAFC; }
        .brand-sub { color:#94A3B8; font-size:0.92rem; margin-top:4px; }
        .nav-pills { display:flex; gap:10px; flex-wrap:wrap; }
        .nav-pill { padding:10px 14px; border-radius:999px; background:#1F2937; color:#CBD5E1; font-size:0.8rem; font-weight:700; }
        .card { background:#111827; border:1px solid rgba(148,163,184,0.10); border-radius:20px; padding:18px; margin-bottom:16px; }
        .section-title { color:#F8FAFC; font-size:1.02rem; font-weight:800; margin-bottom:8px; }
        .section-sub { color:#94A3B8; font-size:0.9rem; margin-bottom:12px; }
        .metric-grid { display:grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap:12px; }
        .metric-card { background:#0F172A; border:1px solid rgba(148,163,184,0.10); border-radius:16px; padding:14px; }
        .metric-label { color:#94A3B8; font-size:0.76rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:5px; }
        .metric-value { color:#F8FAFC; font-size:1.15rem; font-weight:800; }
        .risk-chip { display:inline-block; padding:6px 10px; border-radius:999px; font-size:0.78rem; font-weight:700; }
        .alert-item { display:flex; gap:10px; padding:12px 14px; border-radius:16px; background:#0F172A; border:1px solid rgba(148,163,184,0.08); margin-bottom:10px; }
        .alert-dot { width:10px; height:10px; border-radius:999px; margin-top:6px; }
        .radar-card { background:#0F172A; border:1px solid rgba(148,163,184,0.08); border-radius:18px; padding:16px; margin-bottom:10px; }
        .radar-title { color:#F8FAFC; font-size:1rem; font-weight:800; }
        .radar-meta { color:#94A3B8; font-size:0.85rem; margin-top:4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_topbar() -> None:
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="brand-title">{APP_TITLE}</div>
                <div class="brand-sub">{APP_SUBTITLE}</div>
            </div>
            <div class="nav-pills">
                <div class="nav-pill">Portföy</div>
                <div class="nav-pill">Aksiyon</div>
                <div class="nav-pill">Geçmiş</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def onboarding_sidebar() -> Tuple[str, bool]:
    persisted_ready = get_app_state("onboarding_tamam", "0") == "1"
    persisted_profile = get_app_state("profil", "Dengeli")

    if "onboarding_tamam" not in st.session_state:
        st.session_state["onboarding_tamam"] = persisted_ready
    if "profil" not in st.session_state:
        st.session_state["profil"] = persisted_profile

    st.sidebar.markdown("## Başlangıç")
    if not st.session_state["onboarding_tamam"]:
        s1 = st.sidebar.selectbox("Yatırım süreniz", ["3-6 ay", "6-12 ay", "1 yıl +"], index=1)
        s2 = st.sidebar.selectbox("Portföy %10 düşerse", ["Hemen satarım", "Beklerim", "Alım fırsatı görürüm"], index=1)
        s3 = st.sidebar.selectbox("Yatırım amacınız", ["Sermayeyi korumak", "Dengeli büyüme", "Daha yüksek getiri"], index=1)
        profile_name = "Koruma Odaklı" if s3 == "Sermayeyi korumak" else "Dengeli" if s3 == "Dengeli büyüme" else "Büyüme Odaklı"
        st.sidebar.info(f"Önerilen profil: {profile_name}")
        if st.sidebar.button("Portföyü Başlat", use_container_width=True):
            st.session_state["onboarding_tamam"] = True
            st.session_state["profil"] = profile_name
            set_app_state("onboarding_tamam", "1")
            set_app_state("profil", profile_name)
            if get_app_state("initial_cash", "") == "":
                set_initial_cash(INITIAL_CASH)
            st.rerun()
        return profile_name, False

    profile_name = st.session_state.get("profil", persisted_profile)
    st.sidebar.markdown("## Portföy Menüsü")
    st.sidebar.markdown(f"**Aktif Profil:** {profile_name}")
    st.sidebar.markdown(f"**Başlangıç Bakiye:** {format_price(get_initial_cash())}")
    if st.sidebar.button("Profili Sıfırla", use_container_width=True):
        st.session_state["onboarding_tamam"] = False
        st.session_state["profil"] = "Dengeli"
        set_app_state("onboarding_tamam", "0")
        set_app_state("profil", "Dengeli")
        st.rerun()
    return profile_name, True


def build_open_positions() -> Tuple[List[Dict[str, Any]], float]:
    open_df = load_open_portfolio()
    rows: List[Dict[str, Any]] = []
    invested_now = 0.0
    for _, row in open_df.iterrows():
        symbol = str(row["symbol"])
        raw = download_symbol(symbol, "1G")
        current_price = None
        pnl = 0.0
        pnl_pct = 0.0
        stop = None
        target = None
        action = "TUT"
        alert = "Pozisyon izleniyor"
        value = 0.0
        if not raw.empty:
            df_ind = add_indicators(raw)
            current_price = float(df_ind["Close"].iloc[-1])
            value = current_price * float(row["quantity"])
            invested_now += value
            pnl = (current_price - float(row["entry_price"])) * float(row["quantity"])
            pnl_pct = ((current_price / float(row["entry_price"])) - 1) * 100 if float(row["entry_price"]) else 0.0
            last_atr = safe_float(df_ind["ATR14"].iloc[-1]) or 0.0
            stop = float(row["entry_price"]) - (1.5 * last_atr)
            target = float(row["entry_price"]) + (3.0 * last_atr)
            if current_price <= (stop or -999999):
                action = "ÇIK"
                alert = "Koruyucu stop altı, çıkış değerlendir"
            elif pnl_pct >= 4:
                action = "SATIŞ DÜŞÜN"
                alert = "Hedef bölgesine yaklaşıyor"
            elif pnl_pct <= -3:
                action = "AZALT / DİKKAT"
                alert = "Zarar artıyor, dikkat gerekli"
            else:
                action = "TUT"
                alert = "Yapı korunuyor"
        entry_value = float(row["entry_price"]) * float(row["quantity"])
        rows.append(
            {
                "ID": int(row["id"]),
                "Sembol": symbol,
                "Kaynak": str(row["source"]),
                "Alış": format_price(float(row["entry_price"])),
                "Güncel": format_price(current_price) if current_price is not None else "-",
                "Adet": float(row["quantity"]),
                "Pozisyon Değeri Sayısal": value,
                "Giriş Değeri Sayısal": entry_value,
                "PnL": format_price(pnl),
                "PnL %": f"%{round(float(pnl_pct), 2)}",
                "% Portföy": "-",
                "Durum": action,
                "Stop": format_price(stop) if stop is not None else "-",
                "Hedef": format_price(target) if target is not None else "-",
                "Hedef Pay": f"%{round(float(row.get('target_weight', 0))*100, 0)}" if float(row.get("target_weight", 0)) > 0 else "-",
                "Not": str(row["note"] or ""),
                "Uyarı": alert,
                "PnL Sayısal": pnl,
            }
        )
    return rows, invested_now


def render_portfolio_overview(open_rows: List[Dict[str, Any]], invested_now: float) -> Tuple[float, float, float]:
    initial_cash = get_initial_cash()
    total_pnl = sum(row["PnL Sayısal"] for row in open_rows)
    portfolio_value = initial_cash + total_pnl
    cash = max(0.0, portfolio_value - invested_now)
    riskable = cash * 0.5
    pnl_pct = (total_pnl / max(initial_cash, 1e-9)) * 100
    risk = "DÜŞÜK" if len(open_rows) <= 2 else "ORTA" if len(open_rows) <= 3 else "YÜKSEK"
    risk_color = "#10B981" if risk == "DÜŞÜK" else "#F59E0B" if risk == "ORTA" else "#EF4444"
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Portföy Durumu</div><div class="section-sub">Portföyünün genel görünümü, nakit ve kullanılabilir risk alanı.</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card"><div class="metric-label">Portföy Değeri</div><div class="metric-value">{format_price(portfolio_value)}</div></div>
            <div class="metric-card"><div class="metric-label">Toplam Kar/Zarar</div><div class="metric-value">{format_price(total_pnl)} · %{round(float(pnl_pct), 2)}</div></div>
            <div class="metric-card"><div class="metric-label">Nakit</div><div class="metric-value">{format_price(cash)}</div></div>
            <div class="metric-card"><div class="metric-label">Kullanılabilir Risk</div><div class="metric-value">{format_price(riskable)}</div></div>
        </div>
        <div style="display:flex; justify-content:flex-end; margin-top:10px;"><span class="risk-chip" style="color:{risk_color}; border:1px solid {risk_color};">Portföy Riski: {risk}</span></div>
        """,
        unsafe_allow_html=True,
    )
    if PLOTLY_AVAILABLE:
        hist_points = 14
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=hist_points, freq="D")
        atlas = np.linspace(initial_cash, portfolio_value, hist_points)
        nasdaq = np.linspace(initial_cash, initial_cash * 1.018, hist_points)
        faiz = np.linspace(initial_cash, initial_cash * 1.0035, hist_points)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx, y=atlas, mode="lines", name="Atlas"))
        fig.add_trace(go.Scatter(x=idx, y=nasdaq, mode="lines", name="NASDAQ"))
        fig.add_trace(go.Scatter(x=idx, y=faiz, mode="lines", name="Faiz / PPF"))
        fig.update_layout(height=300, template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#111827", margin=dict(l=10, r=10, t=20, b=10), legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return portfolio_value, cash, riskable


def render_open_positions(rows: List[Dict[str, Any]], portfolio_value: float) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Açık Pozisyonlar</div><div class="section-sub">Mevcut işlemlerini izle, aksiyon önerisini gör ve dilediğinde kapat.</div>', unsafe_allow_html=True)
    if not rows:
        st.info("Henüz açık pozisyon yok.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    for row in rows:
        row["% Portföy"] = f"%{round((row['Pozisyon Değeri Sayısal'] / max(portfolio_value, 1e-9)) * 100, 2)}"
    show_df = pd.DataFrame(rows)[["Sembol", "Kaynak", "Alış", "Güncel", "PnL", "PnL %", "% Portföy", "Hedef Pay", "Durum", "Stop", "Hedef", "Not"]]
    st.dataframe(show_df, use_container_width=True, hide_index=True)
    st.markdown("#### Pozisyon Kapat")
    for row in rows[:5]:
        with st.expander(f"{row['Sembol']} pozisyonunu kapat"):
            exit_price = st.number_input(f"Satış fiyatı · {row['Sembol']}", min_value=0.0, value=0.0, step=0.01, key=f"exit_{row['ID']}")
            if st.button(f"{row['Sembol']} işlemini kapat", key=f"close_{row['ID']}", use_container_width=True):
                if exit_price > 0:
                    close_portfolio_position(int(row["ID"]), float(exit_price))
                    st.success(f"{row['Sembol']} kapatıldı.")
                    st.rerun()
                else:
                    st.warning("Geçerli bir satış fiyatı gir.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_action_center(rows: List[Dict[str, Any]], cash: float, riskable: float, profile_name: str) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Günlük Aksiyon Merkezi</div><div class="section-sub">Portföyüne göre bugün dikkat etmen gerekenler ve yeni öneriler.</div>', unsafe_allow_html=True)
    if not rows:
        st.info("Portföy boşsa önce Atlas başlangıç portföyünden veya yeni işlem ekranından pozisyon ekle.")
    else:
        for item in rows[:3]:
            clr = action_color(item["Durum"])
            st.markdown(
                f"<div class='alert-item'><div class='alert-dot' style='background:{clr};'></div><div><div style='font-weight:800;color:#F8FAFC;'>{item['Sembol']} · {item['Durum']}</div><div style='color:#CBD5E1;'>{item['Uyarı']}</div></div></div>",
                unsafe_allow_html=True,
            )
    st.markdown(f"**Nakit önerisi:** {format_price(cash)} nakdin var. Bunun {format_price(riskable)} kadarı yeni fırsatlar için risk alanı olarak düşünülebilir. Kalan nakit SGOV / PPF tarafında bekleyebilir.")
    radar = build_radar(profile_name, "1G", 3)
    if radar.empty:
        st.info("Bugün profile uygun yeni Atlas işlemi bulunamadı.")
    else:
        st.markdown("#### Yeni Atlas Fırsatları")
        for i, (_, row) in enumerate(radar.iterrows()):
            onerilen_tutar = max(10.0, min(riskable, cash * 0.5 if cash > 0 else 0.0))
            st.markdown(
                f"<div class='radar-card'><div class='radar-title'>{row['Sembol']} · {row['Karar']}</div><div class='radar-meta'>Atlas Skoru: %{row['Atlas Skoru']} · Risk/Getiri: {row['Risk/Getiri']} · Önerilen pozisyon: {format_price(onerilen_tutar)}</div></div>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button(f"{row['Sembol']} formuna aktar", key=f"prepare_{row['Sembol']}_{i}", use_container_width=True):
                    st.session_state["portfolio_prefill_symbol"] = str(row["Sembol"])
                    st.session_state["portfolio_prefill_weight"] = "%15"
                    st.session_state["portfolio_prefill_source"] = "ATLAS TRADE"
                    st.session_state["portfolio_prefill_note"] = f"Atlas fırsatı · Skor %{row['Atlas Skoru']}"
                    st.success(f"{row['Sembol']} yeni işlem formuna aktarıldı.")
            with c2:
                st.caption(f"Stop: {format_price(row['Stop'])} · Hedef: {format_price(row['Hedef'])}")
    st.markdown('</div>', unsafe_allow_html=True)


def render_initial_portfolio_builder(profile_name: str) -> None:
    suggestions = suggest_initial_portfolio(profile_name)
    if "atlas_secili_portfoy" not in st.session_state:
        st.session_state["atlas_secili_portfoy"] = {}

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Atlas Önerilen Başlangıç Portföyü</div><div class="section-sub">Önce hangi varlıkları istediğini seç. Sonra Atlas bunları önerilen oranlarla hazırlasın. Midas\'ta alımı yaptıktan sonra tik atıp portföye ekle.</div>', unsafe_allow_html=True)

    show_df = pd.DataFrame([{k: v for k, v in item.items() if k in ["Varlık", "Tür", "Pay Yazı", "Not"]} for item in suggestions]).rename(columns={"Pay Yazı": "Pay"})
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.markdown("#### 1) Portföy adaylarını seç")
    for idx, item in enumerate(suggestions):
        sembol = item["Varlık"]
        varsayilan = st.session_state["atlas_secili_portfoy"].get(sembol, False)
        secildi = st.checkbox(
            f"{sembol} · {item['Tür']} · {item['Pay Yazı']}",
            value=varsayilan,
            key=f"atlas_select_{sembol}_{idx}",
        )
        st.session_state["atlas_secili_portfoy"][sembol] = secildi
        st.caption(item["Not"])

    secilenler = [item for item in suggestions if st.session_state["atlas_secili_portfoy"].get(item["Varlık"], False)]

    if secilenler:
        st.markdown("#### 2) Seçilen işlemleri hazırla")
        if st.button("Seçilen işlemleri hazırla", use_container_width=True, key="atlas_hazirla_btn"):
            hazir_listesi = []
            for item in secilenler:
                hazir_listesi.append(
                    {
                        "Varlık": item["Varlık"],
                        "Tür": item["Tür"],
                        "Pay": item["Pay"],
                        "Pay Yazı": item["Pay Yazı"],
                        "Not": item["Not"],
                        "Midas Alındı": False,
                        "Alış Fiyatı": 0.0,
                        "Adet": 0.0,
                    }
                )
            st.session_state["atlas_hazir_portfoy"] = hazir_listesi
            st.success("Seçilen varlıklar hazırlandı. Şimdi Midas'ta alım yapıp aşağıdan işlemleri portföye ekleyebilirsin.")
            st.rerun()

    hazir_portfoy = st.session_state.get("atlas_hazir_portfoy", [])
    if hazir_portfoy:
        st.markdown("#### 3) Midas alımını onayla ve portföye ekle")
        baslangic_bakiye = get_initial_cash()
        for idx, item in enumerate(hazir_portfoy):
            sembol = item["Varlık"]
            st.markdown(f"**{sembol}** · {item['Tür']} · Önerilen pay {item['Pay Yazı']}")
            c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1.2])
            with c1:
                midas_alindi = st.checkbox("Midas'ta aldım", value=item.get("Midas Alındı", False), key=f"midas_alindi_{sembol}_{idx}")
            with c2:
                alis_fiyati = st.number_input("Alış fiyatı", min_value=0.0, value=float(item.get("Alış Fiyatı", 0.0)), step=0.01, key=f"alis_fiyati_{sembol}_{idx}")
            with c3:
                varsayilan_tutar = baslangic_bakiye * float(item["Pay"])
                varsayilan_adet = 0.0 if alis_fiyati <= 0 else round(varsayilan_tutar / max(alis_fiyati, 1e-9), 4)
                adet = st.number_input("Adet", min_value=0.0, value=float(item.get("Adet", varsayilan_adet)), step=0.0001, key=f"adet_{sembol}_{idx}")
            with c4:
                pozisyon_degeri = alis_fiyati * adet
                st.markdown(f"Pozisyon değeri: **{format_price(pozisyon_degeri)}**")

            item["Midas Alındı"] = midas_alindi
            item["Alış Fiyatı"] = alis_fiyati
            item["Adet"] = adet

            c5, c6 = st.columns([1, 1])
            with c5:
                if st.button(f"{sembol} portföye ekle", key=f"tekli_portfoy_ekle_{sembol}_{idx}", use_container_width=True):
                    if not midas_alindi:
                        st.warning(f"Önce {sembol} için Midas alımını onayla.")
                    elif alis_fiyati <= 0 or adet <= 0:
                        st.warning(f"{sembol} için alış fiyatı ve adet girmen gerekiyor.")
                    else:
                        add_portfolio_position(
                            sembol,
                            alis_fiyati,
                            adet,
                            f"Atlas önerisi · {item['Not']} · Önerilen pay {item['Pay Yazı']}",
                            "ATLAS TRADE",
                            float(item["Pay"]),
                        )
                        st.success(f"{sembol} portföye eklendi. Atlas artık bu işlemi izleyecek.")
                        st.session_state["atlas_hazir_portfoy"] = [x for x in hazir_portfoy if x["Varlık"] != sembol]
                        st.rerun()
            with c6:
                if st.button(f"{sembol} oranı revize et", key=f"tekli_revize_{sembol}_{idx}", use_container_width=True):
                    st.session_state["portfolio_prefill_symbol"] = sembol
                    st.session_state["portfolio_prefill_weight"] = item["Pay Yazı"]
                    st.session_state["portfolio_prefill_source"] = "ATLAS TRADE"
                    st.session_state["portfolio_prefill_note"] = f"Atlas önerisi · {item['Not']}"
                    st.info(f"{sembol} düzenlenebilir şekilde yeni işlem formuna aktarıldı.")

        if hazir_portfoy:
            st.markdown("#### 4) Toplu işlem")
            if st.button("Midas'ta aldığım seçili işlemleri toplu ekle", use_container_width=True, key="toplu_midas_ekle"):
                eklendi = 0
                kalanlar = []
                for item in hazir_portfoy:
                    if item.get("Midas Alındı") and float(item.get("Alış Fiyatı", 0)) > 0 and float(item.get("Adet", 0)) > 0:
                        add_portfolio_position(
                            item["Varlık"],
                            float(item["Alış Fiyatı"]),
                            float(item["Adet"]),
                            f"Atlas önerisi · {item['Not']} · Önerilen pay {item['Pay Yazı']}",
                            "ATLAS TRADE",
                            float(item["Pay"]),
                        )
                        eklendi += 1
                    else:
                        kalanlar.append(item)
                st.session_state["atlas_hazir_portfoy"] = kalanlar
                if eklendi > 0:
                    st.success(f"{eklendi} işlem portföye eklendi. Atlas artık takip ve yönlendirme yapacak.")
                    st.rerun()
                else:
                    st.warning("Toplu ekleme için önce Midas alımını işaretleyip fiyat ve adet girmen gerekiyor.")

    st.markdown("**Akış:** Önce seç, sonra Midas'ta al, ardından tik atıp fiyat ve adedi girerek portföye ekle. Atlas bundan sonra portföyü izleyip yön vermeye başlar.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_add_position(default_symbol: str) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Yeni İşlem Ekle</div><div class="section-sub">Midas\'ta aldığın varlığı manuel ekle. Atlas bundan sonra takip etsin. İstersen Atlas\'ın önerdiği oranı hızlıca kullanabilirsin.</div>', unsafe_allow_html=True)
    if "portfolio_prefill_symbol" not in st.session_state:
        st.session_state["portfolio_prefill_symbol"] = default_symbol
    if "portfolio_prefill_weight" not in st.session_state:
        st.session_state["portfolio_prefill_weight"] = "%20"
    if "portfolio_prefill_source" not in st.session_state:
        st.session_state["portfolio_prefill_source"] = "KULLANICI"
    if "portfolio_prefill_note" not in st.session_state:
        st.session_state["portfolio_prefill_note"] = "Midas işlemi"
    default_idx = SEARCHABLE_ASSETS.index(st.session_state["portfolio_prefill_symbol"]) if st.session_state["portfolio_prefill_symbol"] in SEARCHABLE_ASSETS else 0
    open_rows, invested_now = build_open_positions()
    portfolio_value = get_initial_cash() + sum(r["PnL Sayısal"] for r in open_rows)

    with st.form("add_position_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            symbol = st.selectbox("Varlık Ara", SEARCHABLE_ASSETS, index=default_idx)
        with c2:
            weight_options = ["%10", "%15", "%20", "%25", "%30", "%35"]
            weight_default = st.session_state["portfolio_prefill_weight"] if st.session_state["portfolio_prefill_weight"] in weight_options else "%20"
            selected_weight = st.selectbox("Portföy Payı", weight_options, index=weight_options.index(weight_default))
        with c3:
            entry = st.number_input("Alış Fiyatı", min_value=0.0, value=0.0, step=0.01)
        with c4:
            qty = st.number_input("Adet", min_value=0.0001, value=1.0, step=0.1)
        c5, c6 = st.columns([1, 1])
        with c5:
            source_options = ["KULLANICI", "ATLAS TRADE"]
            source_default = st.session_state["portfolio_prefill_source"] if st.session_state["portfolio_prefill_source"] in source_options else "KULLANICI"
            source = st.selectbox("Kaynak", source_options, index=source_options.index(source_default))
        with c6:
            st.text_input("Mod", value="Önerilenle hızlı ekle" if source == "ATLAS TRADE" else "Manuel ekleme", disabled=True)
        note = st.text_input("Not", value=st.session_state.get("portfolio_prefill_note", "Midas işlemi"))

        selected_weight_value = float(selected_weight.replace("%", "").replace(",", ".")) / 100.0
        position_value_preview = entry * qty
        portfolio_after = portfolio_value + max(position_value_preview - 0.0, 0.0)
        pct_after = (position_value_preview / max(portfolio_after, 1e-9)) * 100 if position_value_preview > 0 else 0.0

        st.markdown(
            f"**Ön İzleme:** Pozisyon değeri {format_price(position_value_preview)} · İşlem sonrası toplam portföy yaklaşık {format_price(portfolio_after)} · Bu işlemin payı yaklaşık %{round(float(pct_after), 2)} · Hedeflenen pay {selected_weight}"
        )
        if pct_after > 40:
            st.warning("Bu işlem portföyde çok büyük ağırlık oluşturuyor. Oranı veya adedi düşürmek daha sağlıklı olabilir.")
        submitted = st.form_submit_button("Portföye Ekle", use_container_width=True)
        if submitted:
            if symbol and entry > 0 and qty > 0:
                add_portfolio_position(symbol, entry, qty, f"{note} · Önerilen pay {selected_weight}", source, selected_weight_value)
                st.session_state["portfolio_prefill_symbol"] = symbol
                st.session_state["portfolio_prefill_weight"] = selected_weight
                st.session_state["portfolio_prefill_source"] = source
                st.session_state["portfolio_prefill_note"] = note
                if portfolio_after > get_initial_cash():
                    set_initial_cash(portfolio_after)
                st.success(f"{symbol} portföye eklendi. Atlas artık bu işlemi izleyecek.")
                st.rerun()
            else:
                st.warning("Varlık, alış fiyatı ve adet bilgisi gerekli.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_history_tab() -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Geçmiş İşlemler</div><div class="section-sub">Kapatılmış işlemler, gerçekleşen kar/zarar ve işlem geçmişi.</div>', unsafe_allow_html=True)
    hist = load_history(100)
    if hist.empty:
        st.info("Henüz kapatılmış işlem yok.")
    else:
        show = hist.copy()
        for col in ["entry_price", "exit_price", "pnl"]:
            show[col] = show[col].apply(lambda x: format_price(float(x)))
        show["pnl_pct"] = show["pnl_pct"].apply(lambda x: f"%{round(float(x), 2)}")
        st.dataframe(show[["symbol", "source", "entry_price", "exit_price", "pnl", "pnl_pct", "closed_at"]], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


def main_streamlit() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    inject_css()
    init_db()
    render_topbar()
    profile_name, ready = onboarding_sidebar()
    if not ready:
        st.markdown('<div class="card"><div class="section-title">Atlas ile Başlangıç</div><div class="section-sub">3 kısa sorudan sonra sana uygun profil belirlenir. Ardından Atlas sana Türkçe ve sade bir başlangıç portföyü önerir.</div></div>', unsafe_allow_html=True)
        return
    tabs = st.tabs(["👜 Portföy", "➕ Yeni İşlem", "🗃️ Geçmiş"])
    open_rows, invested_now = build_open_positions()
    with tabs[0]:
        if not open_rows:
            render_initial_portfolio_builder(profile_name)
        portfolio_value, cash, riskable = render_portfolio_overview(open_rows, invested_now)
        render_open_positions(open_rows, portfolio_value)
        render_action_center(open_rows, cash, riskable, profile_name)
    with tabs[1]:
        render_add_position(DEFAULT_SYMBOLS[0])
    with tabs[2]:
        render_history_tab()


def print_environment_help() -> None:
    print(APP_TITLE)
    print("Bu dosya Streamlit uygulaması olarak çalışır.")
    if not STREAMLIT_AVAILABLE:
        print("- Eksik paket: streamlit")
    if not YFINANCE_AVAILABLE:
        print("- Eksik paket: yfinance")
    if not PLOTLY_AVAILABLE:
        print("- Eksik paket: plotly")


class TestAtlasMoney(unittest.TestCase):
    def setUp(self) -> None:
        idx = pd.date_range("2025-01-01", periods=80, freq="D")
        base = np.linspace(100, 140, len(idx))
        self.df = pd.DataFrame({
            "Open": base - 0.5,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base,
            "Volume": np.full(len(idx), 1000),
        }, index=idx)

    def test_add_indicators_columns(self):
        out = add_indicators(self.df)
        self.assertIn("EMA200", out.columns)
        self.assertIn("ATR14", out.columns)

    def test_build_risk_levels(self):
        out = add_indicators(self.df)
        sl, tp, rr = build_risk_levels(out, "AL")
        self.assertIsNotNone(sl)
        self.assertIsNotNone(tp)
        self.assertIsNotNone(rr)

    def test_format_price(self):
        self.assertEqual(format_price(1000.5), "1.000,50")

    def test_searchable_assets_contains_safe_assets(self):
        self.assertIn("GLD", SEARCHABLE_ASSETS)
        self.assertIn("SGOV", SEARCHABLE_ASSETS)

    def test_profile_presets(self):
        self.assertIn("Dengeli", PROFILE_PRESETS)

    def test_suggest_initial_portfolio_count(self):
        suggestions = suggest_initial_portfolio("Dengeli")
        self.assertEqual(len(suggestions), 5)


def run_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAtlasMoney)
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
