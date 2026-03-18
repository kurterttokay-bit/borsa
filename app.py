import math
import sqlite3
import sys
import time
import unittest
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

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

APP_TITLE = "Atlas Money V2"
APP_SUBTITLE = "Karar motoru güçlendirilmiş yatırım aracı"
DB_PATH = "signals_v2.db"
DEFAULT_BUY_FEE = 1.50
DEFAULT_SELL_FEE = 1.50
DEFAULT_RISK_PER_TRADE = 0.01
DEFAULT_MAX_POSITIONS = 5

MIDAS_DEMO_DATE = "2026-03-18T09:30:00"
MIDAS_DEMO_INITIAL_TRY = 10000.0
MIDAS_DEMO_USD = 224.05
MIDAS_DEMO_USDTRY = 44.6311
MIDAS_DEMO_POSITIONS = [
    {"symbol": "ROKU", "entry_price": 97.50, "quantity": 0.323609658, "gross_amount": 31.55, "buy_fee": 1.50, "source": "MIDAS DEMO"},
    {"symbol": "NET", "entry_price": 213.13, "quantity": 0.140763123, "gross_amount": 30.00, "buy_fee": 1.50, "source": "MIDAS DEMO"},
    {"symbol": "PANW", "entry_price": 169.76, "quantity": 0.265083236, "gross_amount": 45.00, "buy_fee": 1.50, "source": "MIDAS DEMO"},
    {"symbol": "GOOGL", "entry_price": 308.22, "quantity": 0.243337421, "gross_amount": 75.00, "buy_fee": 1.50, "source": "MIDAS DEMO"},
    {"symbol": "MRVL", "entry_price": 91.27, "quantity": 0.383511209, "gross_amount": 35.00, "buy_fee": 1.50, "source": "MIDAS DEMO"},
]

TIMEFRAME_MAP = {
    "1G": {"interval": "1d", "period": "2y"},
    "3A": {"interval": "1d", "period": "6mo"},
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
SEARCHABLE_ASSETS = sorted(set(RISKY_UNIVERSE + SAFE_ASSETS))
DEFAULT_SYMBOLS = US_MEGA_CAPS.copy()

PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Koruma Odaklı": {"threshold": 67, "risk_label": "Düşük", "max_positions": 4, "risk_per_trade": 0.0075},
    "Dengeli": {"threshold": 62, "risk_label": "Orta", "max_positions": 5, "risk_per_trade": 0.01},
    "Büyüme Odaklı": {"threshold": 58, "risk_label": "Orta", "max_positions": 6, "risk_per_trade": 0.0125},
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
    ema50: float
    ema200: float
    rsi: float
    macd: float
    macd_signal: float
    atr: Optional[float]
    adx_proxy: float
    volume_ratio: float
    market_strength: float
    long_probability: float
    short_probability: float
    verdict: str
    mtf_bias: str
    stop_loss: Optional[float]
    take_profit: Optional[float]
    rr_ratio: Optional[float]
    regime_ok: bool


# ------------------------------
# Database
# ------------------------------

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
            asset_type TEXT DEFAULT 'Hisse',
            coupon_date TEXT DEFAULT '',
            entry_fx REAL DEFAULT 0,
            buy_fee REAL DEFAULT 0,
            gross_amount REAL DEFAULT 0,
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
            source TEXT,
            asset_type TEXT DEFAULT 'Hisse',
            entry_fx REAL DEFAULT 0,
            exit_fx REAL DEFAULT 0,
            pnl_try REAL DEFAULT 0,
            buy_fee REAL DEFAULT 0,
            sell_fee REAL DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cash_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            tx_type TEXT,
            amount REAL,
            note TEXT DEFAULT ''
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dividends_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            symbol TEXT,
            quantity REAL,
            amount_per_share REAL,
            gross_amount REAL,
            withholding_rate REAL DEFAULT 0,
            withholding_tax REAL DEFAULT 0,
            net_amount REAL DEFAULT 0,
            fx_rate REAL DEFAULT 0,
            net_amount_try REAL DEFAULT 0,
            note TEXT DEFAULT ''
        )
        """
    )
    conn.commit()
    conn.close()
    maybe_sync_initial_cash_with_positions()


def get_app_state(key: str, default: str = "") -> str:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
    except sqlite3.OperationalError:
        conn.close()
        return default
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


def get_app_state_float(key: str, default: float = 0.0) -> float:
    raw = get_app_state(key, "")
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


# ------------------------------
# Settings / formatting
# ------------------------------

def get_display_currency() -> str:
    value = get_app_state("display_currency", "USD").upper()
    return value if value in ["USD", "TRY"] else "USD"


def set_display_currency(value: str) -> None:
    set_app_state("display_currency", value.upper())


def get_estimated_tax_rate() -> float:
    value = get_app_state_float("estimated_tax_rate", 15.0)
    if value < 0:
        value = 0.0
    if value > 100:
        value = 100.0
    return float(value)


def set_estimated_tax_rate(value: float) -> None:
    set_app_state("estimated_tax_rate", str(float(max(0.0, min(100.0, value)))))


def format_number(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_try(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"₺{format_number(value)}"


def convert_money(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    currency = get_display_currency()
    if currency == "TRY":
        return float(value) * get_usdtry_rate()
    return float(value)


def format_price(value: Optional[float], is_money: bool = True) -> str:
    if value is None:
        return "-"
    shown = convert_money(value) if is_money else float(value)
    number = format_number(shown)
    if not is_money:
        return number
    currency = get_display_currency()
    return f"${number}" if currency == "USD" else f"₺{number}"


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


# ------------------------------
# Cash / portfolio state
# ------------------------------

def infer_initial_cash() -> float:
    open_df = load_open_portfolio()
    if not open_df.empty:
        gross = pd.to_numeric(open_df["gross_amount"], errors="coerce").fillna(0)
        fees = pd.to_numeric(open_df["buy_fee"], errors="coerce").fillna(0)
        return float((gross + fees).sum())
    hist = load_history(500)
    if not hist.empty:
        entry_cost = (
            pd.to_numeric(hist["entry_price"], errors="coerce").fillna(0)
            * pd.to_numeric(hist["quantity"], errors="coerce").fillna(0)
        )
        fees = pd.to_numeric(hist.get("buy_fee", 0), errors="coerce").fillna(0)
        return float((entry_cost + fees).max())
    return 0.0


def get_initial_cash() -> float:
    raw = get_app_state("initial_cash", "")
    if raw != "":
        try:
            return float(raw)
        except Exception:
            pass
    return infer_initial_cash()


def set_initial_cash(value: float, source: str = "manual") -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO app_state (key, value) VALUES ('initial_cash', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(float(value)),),
    )
    conn.execute(
        "INSERT INTO app_state (key, value) VALUES ('initial_cash_source', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (source,),
    )
    conn.commit()
    conn.close()


def maybe_sync_initial_cash_with_positions() -> None:
    if get_app_state("initial_cash_source", "") == "manual":
        return
    if not load_cash_transactions().empty:
        return
    if not load_history(1).empty:
        return
    if get_app_state("demo_seeded", "0") == "1":
        return
    inferred = infer_initial_cash()
    if inferred > 0:
        set_initial_cash(inferred, source="auto")


def add_cash_transaction(tx_type: str, amount: float, note: str = "", created_at: Optional[str] = None) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO cash_transactions (created_at, tx_type, amount, note)
        VALUES (?, ?, ?, ?)
        """,
        (
            created_at or datetime.now().isoformat(timespec="seconds"),
            tx_type,
            float(amount),
            note,
        ),
    )
    conn.commit()
    conn.close()


def load_cash_transactions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM cash_transactions ORDER BY id DESC", conn)
    except Exception:
        df = pd.DataFrame(columns=["id", "created_at", "tx_type", "amount", "note"])
    conn.close()
    return df


def add_dividend_record(symbol: str, quantity: float, amount_per_share: float, withholding_rate: float = 0.0, note: str = "") -> None:
    gross_amount = float(quantity) * float(amount_per_share)
    withholding_tax = gross_amount * (float(withholding_rate) / 100.0)
    net_amount = gross_amount - withholding_tax
    fx_rate = get_usdtry_rate()
    net_amount_try = net_amount * fx_rate
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO dividends_history (
            created_at, symbol, quantity, amount_per_share, gross_amount,
            withholding_rate, withholding_tax, net_amount, fx_rate, net_amount_try, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            symbol,
            float(quantity),
            float(amount_per_share),
            float(gross_amount),
            float(withholding_rate),
            float(withholding_tax),
            float(net_amount),
            float(fx_rate),
            float(net_amount_try),
            note,
        ),
    )
    conn.commit()
    conn.close()


def load_dividend_history() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM dividends_history ORDER BY id DESC", conn)
    except Exception:
        df = pd.DataFrame(columns=["id", "created_at", "symbol", "quantity", "amount_per_share", "gross_amount", "withholding_rate", "withholding_tax", "net_amount", "fx_rate", "net_amount_try", "note"])
    conn.close()
    return df


def get_dividend_summary() -> Dict[str, float]:
    div_df = load_dividend_history()
    if div_df.empty:
        return {"gross_dividends": 0.0, "withholding_tax": 0.0, "net_dividends": 0.0, "net_dividends_try": 0.0}
    gross_dividends = float(pd.to_numeric(div_df["gross_amount"], errors="coerce").fillna(0).sum())
    withholding_tax = float(pd.to_numeric(div_df["withholding_tax"], errors="coerce").fillna(0).sum())
    net_dividends = float(pd.to_numeric(div_df["net_amount"], errors="coerce").fillna(0).sum())
    net_dividends_try = float(pd.to_numeric(div_df["net_amount_try"], errors="coerce").fillna(0).sum())
    return {
        "gross_dividends": gross_dividends,
        "withholding_tax": withholding_tax,
        "net_dividends": net_dividends,
        "net_dividends_try": net_dividends_try,
    }


def get_cash_flow_summary() -> Dict[str, float]:
    cash_df = load_cash_transactions()
    deposits = 0.0
    withdrawals = 0.0
    if not cash_df.empty:
        deposits = float(cash_df.loc[cash_df["tx_type"] == "deposit", "amount"].sum())
        withdrawals = float(cash_df.loc[cash_df["tx_type"] == "withdrawal", "amount"].sum())
    net_capital = get_initial_cash() + deposits - withdrawals
    return {
        "deposits": deposits,
        "withdrawals": withdrawals,
        "net_capital": net_capital,
    }


def add_portfolio_position(
    symbol: str,
    entry_price: float,
    quantity: float,
    note: str = "",
    source: str = "KULLANICI",
    target_weight: float = 0.0,
    asset_type: str = "Hisse",
    coupon_date: str = "",
    entry_fx: Optional[float] = None,
    buy_fee: float = 0.0,
    gross_amount: Optional[float] = None,
    created_at: Optional[str] = None,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    effective_entry_fx = float(entry_fx) if entry_fx and entry_fx > 0 else float(get_usdtry_rate())
    cur.execute(
        """
        INSERT INTO portfolio (
            created_at, symbol, entry_price, quantity, note, source, target_weight,
            asset_type, coupon_date, status, entry_fx, buy_fee, gross_amount
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
        """,
        (
            created_at or datetime.now().isoformat(timespec="seconds"),
            symbol,
            float(entry_price),
            float(quantity),
            note,
            source,
            float(target_weight),
            asset_type,
            coupon_date,
            effective_entry_fx,
            float(buy_fee),
            float(gross_amount) if gross_amount is not None else float(entry_price) * float(quantity),
        ),
    )
    conn.commit()
    conn.close()


def load_open_portfolio() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM portfolio WHERE status='open' ORDER BY id DESC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def load_history(limit: int = 100) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(f"SELECT * FROM trades_history ORDER BY id DESC LIMIT {int(limit)}", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def close_portfolio_position(position_id: int, exit_price: float, sell_fee: float = DEFAULT_SELL_FEE) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    row = cur.execute(
        "SELECT created_at, symbol, entry_price, quantity, note, source, asset_type, COALESCE(entry_fx, 0), COALESCE(buy_fee, 0) FROM portfolio WHERE id = ?",
        (int(position_id),),
    ).fetchone()
    if row is None:
        conn.close()
        return
    opened_at, symbol, entry_price, quantity, note, source, asset_type, entry_fx, buy_fee = row
    gross_pnl = (float(exit_price) - float(entry_price)) * float(quantity)
    pnl = gross_pnl - float(buy_fee or 0) - float(sell_fee or 0)
    entry_cost = (float(entry_price) * float(quantity)) + float(buy_fee or 0)
    pnl_pct = (pnl / max(entry_cost, 1e-9)) * 100
    effective_entry_fx = float(entry_fx) if float(entry_fx or 0) > 0 else float(get_usdtry_rate())
    exit_fx = float(get_usdtry_rate())
    pnl_try = (
        (float(exit_price) * exit_fx * float(quantity))
        - (float(entry_price) * effective_entry_fx * float(quantity))
        - (float(buy_fee or 0) * effective_entry_fx)
        - (float(sell_fee or 0) * exit_fx)
    )
    cur.execute(
        """
        INSERT INTO trades_history (
            opened_at, closed_at, symbol, entry_price, exit_price, quantity, pnl, pnl_pct,
            note, source, asset_type, entry_fx, exit_fx, pnl_try, buy_fee, sell_fee
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            asset_type,
            effective_entry_fx,
            exit_fx,
            float(pnl_try),
            float(buy_fee or 0),
            float(sell_fee or 0),
        ),
    )
    cur.execute("UPDATE portfolio SET status='closed' WHERE id = ?", (int(position_id),))
    conn.commit()
    conn.close()


def has_any_user_data() -> bool:
    return (not load_open_portfolio().empty) or (not load_history(1).empty) or (not load_cash_transactions().empty)


def seed_midas_demo_portfolio(force: bool = False) -> None:
    if get_app_state("demo_seeded", "0") == "1" and not force:
        return
    if force:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM portfolio")
        conn.execute("DELETE FROM trades_history")
        conn.execute("DELETE FROM cash_transactions")
        conn.execute("DELETE FROM dividends_history")
        conn.commit()
        conn.close()
    set_app_state("manual_usdtry", str(MIDAS_DEMO_USDTRY))
    set_initial_cash(MIDAS_DEMO_USD, source="manual")
    set_app_state("demo_seed_label", f"Midas demo · {MIDAS_DEMO_DATE}")
    for p in MIDAS_DEMO_POSITIONS:
        add_portfolio_position(
            symbol=p["symbol"],
            entry_price=float(p["entry_price"]),
            quantity=float(p["quantity"]),
            source=str(p.get("source", "MIDAS DEMO")),
            gross_amount=float(p["gross_amount"]),
            buy_fee=float(p.get("buy_fee", DEFAULT_BUY_FEE)),
            created_at=MIDAS_DEMO_DATE,
            note="Demo başlangıç portföyü",
        )
    set_app_state("demo_seeded", "1")


# ------------------------------
# Market data / indicators
# ------------------------------
@cache_data(ttl=900, show_spinner=False)
def get_usdtry_rate() -> float:
    if YFINANCE_AVAILABLE:
        try:
            df = yf.download("USDTRY=X", period="5d", interval="1d", progress=False, threads=False, auto_adjust=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                close = pd.to_numeric(df["Close"], errors="coerce").dropna()
                if not close.empty:
                    return float(close.iloc[-1])
        except Exception:
            pass
    fallback = get_app_state_float("manual_usdtry", 0.0)
    return float(fallback) if fallback > 0 else 1.0


@cache_data(ttl=300, show_spinner=False)
def get_latest_symbol_price(symbol: str) -> Optional[float]:
    if not YFINANCE_AVAILABLE:
        return None
    try:
        df = yf.download(symbol, period="5d", interval="1d", progress=False, threads=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if close.empty:
            return None
        return float(close.iloc[-1])
    except Exception:
        return None


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


def adx_proxy(df: pd.DataFrame, length: int = 14) -> pd.Series:
    rolling = df["Close"].pct_change().rolling(length)
    signal = (rolling.mean().abs() / rolling.std().replace(0, np.nan)) * 25
    return signal.clip(lower=5, upper=50).fillna(15)


@cache_data(ttl=180, show_spinner=False)
def download_symbol(symbol: str, tf: str) -> pd.DataFrame:
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()
    cfg = TIMEFRAME_MAP[tf]
    try:
        time.sleep(0.05)
        df = yf.download(
            tickers=symbol,
            interval=cfg["interval"],
            period=cfg["period"],
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.title)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in df.columns for col in required):
        return pd.DataFrame()
    clean = df[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    return clean if not clean.empty else pd.DataFrame()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA21"] = ema(out["Close"], 21)
    out["EMA50"] = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)
    out["RSI14"] = rsi(out["Close"], 14)
    out["MACD"], out["MACD_SIGNAL"] = macd(out["Close"])
    out["ATR14"] = atr(out, 14)
    out["VOL20"] = out["Volume"].rolling(20).mean()
    out["VOL_RATIO"] = (out["Volume"] / out["VOL20"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out["RET20"] = out["Close"].pct_change(20)
    out["ADX_PROXY"] = adx_proxy(out, 14)
    return out


def build_risk_levels(df: pd.DataFrame, verdict: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    last = df.iloc[-1]
    close = float(last["Close"])
    atr_value = safe_float(last["ATR14"])
    if not atr_value or atr_value <= 0:
        return None, None, None
    if verdict in ["SAT", "GÜÇLÜ SAT"]:
        sl = close + (1.8 * atr_value)
        tp = close - (3.2 * atr_value)
        rr = (close - tp) / max(sl - close, 1e-9)
    else:
        sl = close - (1.8 * atr_value)
        tp = close + (3.2 * atr_value)
        rr = (tp - close) / max(close - sl, 1e-9)
    return round(sl, 4), round(tp, 4), round(rr, 2)


def get_market_regime() -> Dict[str, Any]:
    raw = download_symbol("SPY", "1G")
    if raw.empty:
        return {"label": "NÖTR", "risk_multiplier": 0.75, "trend_ok": True}
    df = add_indicators(raw)
    last = df.iloc[-1]
    trend_ok = bool(last["Close"] > last["EMA200"] and last["EMA50"] > last["EMA200"])
    rsi_ok = bool(last["RSI14"] >= 48)
    ret_ok = bool(last["RET20"] > -0.02)
    if trend_ok and rsi_ok and ret_ok:
        return {"label": "RISK-ON", "risk_multiplier": 1.0, "trend_ok": True}
    if (last["Close"] > last["EMA200"]) or rsi_ok:
        return {"label": "NÖTR", "risk_multiplier": 0.65, "trend_ok": True}
    return {"label": "RISK-OFF", "risk_multiplier": 0.35, "trend_ok": False}


def analyze_symbol(symbol: str, timeframe: str = "1G") -> Tuple[pd.DataFrame, Optional[AnalysisResult], Optional[str]]:
    try:
        raw = download_symbol(symbol, timeframe)
        if raw.empty:
            return pd.DataFrame(), None, "Veri alınamadı"
        df = add_indicators(raw)
        if len(df) < 50:
            return pd.DataFrame(), None, "Yetersiz veri"
        last = df.iloc[-1]
        if pd.isna(last["Close"]) or pd.isna(last["EMA200"]) or pd.isna(last["RSI14"]):
            return pd.DataFrame(), None, "Eksik gösterge verisi"
        score = 50.0
        close = float(last["Close"])
        ema50v = float(last["EMA50"])
        ema200v = float(last["EMA200"])
        rsi_v = float(last["RSI14"])
        macd_v = float(last["MACD"])
        macd_sig = float(last["MACD_SIGNAL"])
        vol_ratio = float(last["VOL_RATIO"])
        trend_strength = float(last["ADX_PROXY"])
        ret20 = float(last["RET20"] or 0)

        if close > ema200v:
            score += 15
        else:
            score -= 15
        if ema50v > ema200v:
            score += 10
        else:
            score -= 8
        if close > ema50v:
            score += 8
        if 52 <= rsi_v <= 67:
            score += 10
        elif 45 <= rsi_v < 52:
            score += 4
        elif rsi_v < 40:
            score -= 10
        elif rsi_v > 74:
            score -= 6
        if macd_v > macd_sig:
            score += 10
        else:
            score -= 7
        if vol_ratio > 1.1:
            score += 6
        elif vol_ratio < 0.85:
            score -= 4
        if trend_strength > 22:
            score += 7
        if ret20 > 0.05:
            score += 6
        elif ret20 < -0.05:
            score -= 8

        regime = get_market_regime()
        if not regime["trend_ok"] and symbol in RISKY_UNIVERSE:
            score -= 10

        score = max(0, min(100, score))
        if score >= 78:
            verdict = "GÜÇLÜ AL"
        elif score >= 62:
            verdict = "AL"
        elif score <= 24:
            verdict = "GÜÇLÜ SAT"
        elif score <= 40:
            verdict = "SAT"
        else:
            verdict = "İZLE"
        sl, tp, rr = build_risk_levels(df, verdict)
        mtf_bias = "UYUMLU YUKARI" if close > ema200v else "UYUMLU AŞAĞI"
        regime_ok = regime["trend_ok"] or symbol in SAFE_ASSETS
        result = AnalysisResult(
            symbol=symbol,
            timeframe=timeframe,
            close=round(close, 4),
            ema50=round(ema50v, 4),
            ema200=round(ema200v, 4),
            rsi=round(rsi_v, 2),
            macd=round(macd_v, 4),
            macd_signal=round(macd_sig, 4),
            atr=round(float(last["ATR14"]), 4) if not pd.isna(last["ATR14"]) else None,
            adx_proxy=round(trend_strength, 2),
            volume_ratio=round(vol_ratio, 2),
            market_strength=round(score, 1),
            long_probability=round(score, 1),
            short_probability=round(100 - score, 1),
            verdict=verdict,
            mtf_bias=mtf_bias,
            stop_loss=sl,
            take_profit=tp,
            rr_ratio=rr,
            regime_ok=regime_ok,
        )
        return df, result, None
    except Exception:
        return pd.DataFrame(), None, "Analiz sırasında hata oluştu"


def estimate_pairwise_correlation(symbol: str, open_symbols: List[str]) -> float:
    if not open_symbols:
        return 0.0
    target = download_symbol(symbol, "3A")
    if target.empty:
        return 0.0
    target_ret = target["Close"].pct_change().dropna()
    corrs: List[float] = []
    for s in open_symbols[:8]:
        if s == symbol:
            continue
        other = download_symbol(s, "3A")
        if other.empty:
            continue
        other_ret = other["Close"].pct_change().dropna()
        merged = pd.concat([target_ret, other_ret], axis=1, join="inner").dropna()
        if len(merged) < 20:
            continue
        corr = merged.iloc[:, 0].corr(merged.iloc[:, 1])
        if pd.notna(corr):
            corrs.append(float(corr))
    return float(np.mean(corrs)) if corrs else 0.0


def suggest_position_size(
    cash: float,
    current_equity: float,
    entry_price: float,
    stop_loss: Optional[float],
    risk_per_trade: float,
    buy_fee: float = DEFAULT_BUY_FEE,
    regime_multiplier: float = 1.0,
    correlation_penalty: float = 0.0,
    max_weight: float = 0.20,
) -> Dict[str, float]:
    if stop_loss is None or stop_loss >= entry_price:
        return {"shares": 0.0, "gross_amount": 0.0, "risk_amount": 0.0, "target_weight": 0.0}
    per_share_risk = max(entry_price - stop_loss, 0.01)
    risk_budget = max(current_equity, 0.0) * max(risk_per_trade, 0.001) * max(regime_multiplier, 0.1)
    penalty_factor = max(0.4, 1.0 - max(0.0, correlation_penalty) * 0.35)
    risk_budget *= penalty_factor
    shares_by_risk = risk_budget / per_share_risk
    shares_by_cash = max(0.0, (cash - buy_fee) / max(entry_price, 1e-9))
    shares_by_weight = (current_equity * max_weight) / max(entry_price, 1e-9)
    shares = max(0.0, min(shares_by_risk, shares_by_cash, shares_by_weight))
    gross_amount = shares * entry_price
    target_weight = gross_amount / max(current_equity, 1e-9)
    return {
        "shares": float(max(0.0, shares)),
        "gross_amount": float(max(0.0, gross_amount)),
        "risk_amount": float(max(0.0, shares * per_share_risk)),
        "target_weight": float(max(0.0, target_weight)),
    }


@cache_data(ttl=240, show_spinner=False)
def build_radar(profile_name: str, timeframe: str = "1G", limit: int = 8) -> pd.DataFrame:
    threshold = PROFILE_PRESETS[profile_name]["threshold"]
    regime = get_market_regime()
    rows = []
    for symbol in RISKY_UNIVERSE:
        _, result, err = analyze_symbol(symbol, timeframe)
        if err or result is None:
            continue
        if result.market_strength < threshold:
            continue
        if result.verdict not in ["GÜÇLÜ AL", "AL"]:
            continue
        if not result.regime_ok:
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
                "RSI": result.rsi,
                "Hacim Gücü": result.volume_ratio,
                "Trend Gücü": result.adx_proxy,
                "Rejim": regime["label"],
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["Atlas Skoru", "Trend Gücü", "Hacim Gücü", "Risk/Getiri"], ascending=[False, False, False, False])
        .head(limit)
        .reset_index(drop=True)
    )


def suggest_initial_portfolio(profile_name: str) -> List[Dict[str, Any]]:
    radar = build_radar(profile_name, "1G", 6)
    stock_rows: List[Dict[str, Any]] = []
    if not radar.empty:
        for _, row in radar.head(3).iterrows():
            stock_rows.append(
                {
                    "Varlık": str(row["Sembol"]),
                    "Tür": "Hisse",
                    "Pay": 0.18,
                    "Pay Yazı": "%18",
                    "Not": f"Atlas %{row['Atlas Skoru']} · R/G {row['Risk/Getiri']}",
                }
            )
    while len(stock_rows) < 3:
        fallback = DEFAULT_SYMBOLS[len(stock_rows)]
        stock_rows.append(
            {
                "Varlık": fallback,
                "Tür": "Hisse",
                "Pay": 0.18,
                "Pay Yazı": "%18",
                "Not": "Başlangıç için likit aday",
            }
        )
    return stock_rows + [
        {"Varlık": "GLD", "Tür": "Altın ETF", "Pay": 0.23, "Pay Yazı": "%23", "Not": "Defansif denge"},
        {"Varlık": "SGOV", "Tür": "PPF / Nakit Park", "Pay": 0.23, "Pay Yazı": "%23", "Not": "Fırsat sermayesi"},
    ]


# ------------------------------
# Portfolio intelligence
# ------------------------------

def build_open_positions() -> Tuple[List[Dict[str, Any]], float]:
    open_df = load_open_portfolio()
    rows: List[Dict[str, Any]] = []
    invested_now = 0.0
    for _, row in open_df.iterrows():
        symbol = str(row["symbol"])
        raw = download_symbol(symbol, "1G")
        current_price = None
        gross_pnl = 0.0
        net_pnl = 0.0
        pnl_pct = 0.0
        stop = None
        target = None
        action = "TUT"
        alert = "Pozisyon izleniyor"
        value = 0.0
        buy_fee = float(row.get("buy_fee", 0) or 0)
        gross_amount = float(row.get("gross_amount", 0) or (float(row["entry_price"]) * float(row["quantity"])))
        if not raw.empty:
            df_ind = add_indicators(raw)
            last = df_ind.iloc[-1]
            current_price = float(last["Close"])
            value = current_price * float(row["quantity"])
            invested_now += value
            gross_pnl = (current_price - float(row["entry_price"])) * float(row["quantity"])
            net_pnl = gross_pnl - buy_fee
            net_cost = gross_amount + buy_fee
            pnl_pct = (net_pnl / max(net_cost, 1e-9)) * 100
            stop = float(last["Close"]) - (1.8 * float(last["ATR14"]))
            target = float(last["Close"]) + (3.2 * float(last["ATR14"]))
            if current_price <= (stop or -999999):
                action = "ÇIK"
                alert = "Stop altında, pozisyonu kapatmayı değerlendir"
            elif pnl_pct >= 8:
                action = "KÂR AL / AZALT"
                alert = "Kâr güçlü. Kısmi realize düşünebilirsin"
            elif pnl_pct <= -4:
                action = "AZALT / DİKKAT"
                alert = "Zayıflama var, pozisyon riski arttı"
        entry_value = gross_amount + buy_fee
        fee_pct = (buy_fee / max(gross_amount, 1e-9)) * 100 if gross_amount > 0 else 0.0
        rows.append(
            {
                "ID": int(row["id"]),
                "Sembol": symbol,
                "Kaynak": str(row["source"]),
                "Varlık Türü": str(row.get("asset_type", "Hisse")),
                "Kupon Tarihi": str(row.get("coupon_date", "") or ""),
                "Alış": format_price(float(row["entry_price"])),
                "Güncel": format_price(current_price) if current_price is not None else "-",
                "Adet": float(row["quantity"]),
                "Pozisyon Değeri Sayısal": value,
                "Giriş Değeri Sayısal": entry_value,
                "Brüt PnL": format_price(gross_pnl),
                "Net PnL": format_price(net_pnl),
                "PnL": format_price(net_pnl),
                "PnL %": f"%{round(float(pnl_pct), 2)}",
                "Komisyon": format_price(buy_fee),
                "Komisyon %": f"%{round(fee_pct, 2)}",
                "% Portföy": "-",
                "Durum": action,
                "Stop": format_price(stop) if stop is not None else "-",
                "Hedef": format_price(target) if target is not None else "-",
                "Hedef Pay": f"%{round(float(row.get('target_weight', 0))*100, 0)}" if float(row.get("target_weight", 0)) > 0 else "-",
                "Alım Tutarı": format_price(gross_amount),
                "Hedef Pay Sayısal": float(row.get("target_weight", 0) or 0),
                "Not": str(row["note"] or ""),
                "Uyarı": alert,
                "Brüt PnL Sayısal": gross_pnl,
                "PnL Sayısal": net_pnl,
                "Komisyon Sayısal": buy_fee,
                "Mevcut Pay Sayısal": 0.0,
                "Entry Price Numeric": float(row["entry_price"]),
            }
        )
    return rows, invested_now


def calculate_estimated_tax(history_df: pd.DataFrame) -> Dict[str, float]:
    if history_df.empty:
        return {"taxable_profit_try": 0.0, "estimated_tax_try": 0.0, "tax_rate": get_estimated_tax_rate()}
    taxable_profit_try = float(pd.to_numeric(history_df.get("pnl_try", 0), errors="coerce").fillna(0).clip(lower=0).sum())
    tax_rate = get_estimated_tax_rate()
    estimated_tax_try = taxable_profit_try * (tax_rate / 100.0)
    return {"taxable_profit_try": taxable_profit_try, "estimated_tax_try": estimated_tax_try, "tax_rate": tax_rate}


def compute_portfolio_snapshot(open_rows: List[Dict[str, Any]], invested_now: float) -> Dict[str, float]:
    flow = get_cash_flow_summary()
    net_capital = flow["net_capital"]
    unrealized_pnl = float(sum(r["PnL Sayısal"] for r in open_rows)) if open_rows else 0.0
    gross_unrealized_pnl = float(sum(r.get("Brüt PnL Sayısal", r["PnL Sayısal"]) for r in open_rows)) if open_rows else 0.0
    open_fees = float(sum(r.get("Komisyon Sayısal", 0) or 0 for r in open_rows)) if open_rows else 0.0
    realized_pnl = 0.0
    realized_fees = 0.0
    history_df = load_history(500)
    if not history_df.empty:
        realized_pnl = float(pd.to_numeric(history_df["pnl"], errors="coerce").fillna(0).sum())
        realized_fees = float(pd.to_numeric(history_df.get("buy_fee", 0), errors="coerce").fillna(0).sum())
        realized_fees += float(pd.to_numeric(history_df.get("sell_fee", 0), errors="coerce").fillna(0).sum())
    dividend_summary = get_dividend_summary()
    net_dividends = float(dividend_summary["net_dividends"])
    portfolio_value = invested_now
    current_equity = net_capital + unrealized_pnl + realized_pnl + net_dividends
    open_cost = float(sum(r["Giriş Değeri Sayısal"] for r in open_rows)) if open_rows else 0.0
    cash = max(0.0, current_equity - portfolio_value)
    if get_app_state("initial_cash", "") == "" and open_cost > net_capital and flow["deposits"] == 0 and flow["withdrawals"] == 0 and history_df.empty:
        net_capital = open_cost
        current_equity = net_capital + unrealized_pnl + net_dividends
        cash = max(0.0, current_equity - portfolio_value)
    tax_summary = calculate_estimated_tax(history_df)
    regime = get_market_regime()
    riskable = cash * 0.5 * regime["risk_multiplier"]
    total_fees = open_fees + realized_fees
    gross_total_pnl = gross_unrealized_pnl + realized_pnl + net_dividends
    total_pnl = current_equity - net_capital
    total_return_pct = (total_pnl / max(net_capital, 1e-9)) * 100 if net_capital > 0 else 0.0
    after_tax_total_pnl_usd = total_pnl - (tax_summary["estimated_tax_try"] / max(get_usdtry_rate(), 1e-9))
    after_tax_return_pct = (after_tax_total_pnl_usd / max(net_capital, 1e-9)) * 100 if net_capital > 0 else 0.0
    for row in open_rows:
        row["Mevcut Pay Sayısal"] = row["Pozisyon Değeri Sayısal"] / max(current_equity, 1e-9)
    return {
        "net_capital": net_capital,
        "portfolio_value": portfolio_value,
        "current_equity": current_equity,
        "cash": cash,
        "riskable": riskable,
        "gross_unrealized_pnl": gross_unrealized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "realized_pnl": realized_pnl,
        "dividend_income": net_dividends,
        "gross_total_pnl": gross_total_pnl,
        "total_fees": total_fees,
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "after_tax_total_pnl": after_tax_total_pnl_usd,
        "after_tax_return_pct": after_tax_return_pct,
        "taxable_profit_try": tax_summary["taxable_profit_try"],
        "estimated_tax_try": tax_summary["estimated_tax_try"],
        "estimated_tax_rate": tax_summary["tax_rate"],
        "deposits": flow["deposits"],
        "withdrawals": flow["withdrawals"],
        "market_regime": regime["label"],
        "regime_multiplier": regime["risk_multiplier"],
    }


def get_portfolio_risk_summary(open_rows: List[Dict[str, Any]], current_equity: float) -> Dict[str, float]:
    open_symbols = [r["Sembol"] for r in open_rows]
    if not open_rows:
        return {"largest_weight": 0.0, "avg_corr": 0.0, "diversification_score": 100.0}
    weights = [r["Pozisyon Değeri Sayısal"] / max(current_equity, 1e-9) for r in open_rows]
    largest_weight = max(weights) if weights else 0.0
    corr_vals = []
    for sym in open_symbols[:5]:
        corr = estimate_pairwise_correlation(sym, [s for s in open_symbols if s != sym])
        if corr:
            corr_vals.append(max(corr, 0.0))
    avg_corr = float(np.mean(corr_vals)) if corr_vals else 0.0
    diversification_score = max(0.0, min(100.0, 100.0 - (largest_weight * 100 * 0.8) - (avg_corr * 35)))
    return {"largest_weight": largest_weight, "avg_corr": avg_corr, "diversification_score": diversification_score}


def build_trade_ideas(profile_name: str, open_rows: List[Dict[str, Any]], snapshot: Dict[str, float]) -> pd.DataFrame:
    radar = build_radar(profile_name, "1G", 12)
    if radar.empty:
        return pd.DataFrame()
    open_symbols = [row["Sembol"] for row in open_rows]
    risk_per_trade = PROFILE_PRESETS[profile_name]["risk_per_trade"]
    rows = []
    for _, item in radar.iterrows():
        symbol = str(item["Sembol"])
        corr = estimate_pairwise_correlation(symbol, open_symbols)
        sizing = suggest_position_size(
            cash=snapshot["cash"],
            current_equity=snapshot["current_equity"],
            entry_price=float(item["Fiyat"]),
            stop_loss=safe_float(item["Stop"]),
            risk_per_trade=risk_per_trade,
            regime_multiplier=float(snapshot["regime_multiplier"]),
            correlation_penalty=max(corr, 0.0),
        )
        if sizing["gross_amount"] <= 0:
            continue
        action = "ARTIR" if symbol in open_symbols else "YENİ POZİSYON"
        rows.append(
            {
                "Sembol": symbol,
                "Aksiyon": action,
                "Atlas Skoru": float(item["Atlas Skoru"]),
                "Son Fiyat": float(item["Fiyat"]),
                "Stop": safe_float(item["Stop"]),
                "Hedef": safe_float(item["Hedef"]),
                "Risk/Getiri": safe_float(item["Risk/Getiri"]),
                "Tahmini Korelasyon": round(corr, 2),
                "Önerilen Tutar": round(sizing["gross_amount"], 2),
                "Önerilen Adet": round(sizing["shares"], 4),
                "Risk Bütçesi": round(sizing["risk_amount"], 2),
                "Hedef Pay": round(sizing["target_weight"] * 100, 2),
                "Mesaj": f"{action} · korelasyon {round(corr,2)} · rejim {snapshot['market_regime']}",
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Atlas Skoru", "Tahmini Korelasyon", "Risk/Getiri"], ascending=[False, True, False]).reset_index(drop=True)


# ------------------------------
# Streamlit UI
# ------------------------------

def inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] { font-family: Inter, system-ui, sans-serif; }
        .stApp { background:#0B1220; color:#E5E7EB; }
        .block-container { max-width: 1240px; padding-top: 0.8rem; padding-bottom: 2rem; }
        section[data-testid="stSidebar"] { background:#0F172A; }
        .card { background:#111827; border:1px solid rgba(148,163,184,0.12); border-radius:16px; padding:16px; margin-bottom:12px; }
        .section-title { color:#F8FAFC; font-size:1rem; font-weight:800; margin-bottom:4px; }
        .section-sub { color:#94A3B8; font-size:0.86rem; margin-bottom:12px; }
        .metric-grid { display:grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap:10px; }
        .metric-card { background:#0F172A; border:1px solid rgba(148,163,184,0.08); border-radius:14px; padding:13px 14px; min-height:84px; }
        .metric-label { color:#94A3B8; font-size:0.70rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px; }
        .metric-value { color:#F8FAFC; font-size:1.12rem; font-weight:800; }
        .badge { display:inline-flex; padding:6px 10px; border-radius:999px; background:#0F172A; border:1px solid rgba(148,163,184,0.12); margin-right:8px; color:#CBD5E1; }
        @media (max-width: 900px) { .metric-grid { grid-template-columns: 1fr 1fr; } }
        @media (max-width: 640px) { .metric-grid { grid-template-columns: 1fr; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_topbar(snapshot: Optional[Dict[str, float]] = None) -> None:
    regime = snapshot["market_regime"] if snapshot else get_market_regime()["label"]
    st.markdown(
        f"""
        <div class="card">
            <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;">
                <div>
                    <div style="font-size:1.2rem;font-weight:800;color:#F8FAFC;">{APP_TITLE}</div>
                    <div style="color:#94A3B8;">{APP_SUBTITLE}</div>
                </div>
                <div>
                    <span class="badge">Rejim: {regime}</span>
                    <span class="badge">Kur: 1 USD ≈ ₺{format_number(get_usdtry_rate())}</span>
                </div>
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
        amac = st.sidebar.selectbox("Yatırım amacın", ["Sermayeyi korumak", "Dengeli büyüme", "Daha yüksek getiri"], index=1)
        profile_name = "Koruma Odaklı" if amac == "Sermayeyi korumak" else "Dengeli" if amac == "Dengeli büyüme" else "Büyüme Odaklı"
        st.sidebar.info(f"Önerilen profil: {profile_name}")
        if st.sidebar.button("Atlas V2'yi Başlat", use_container_width=True):
            st.session_state["onboarding_tamam"] = True
            st.session_state["profil"] = profile_name
            set_app_state("onboarding_tamam", "1")
            set_app_state("profil", profile_name)
            st.rerun()
        return profile_name, False
    profile_name = st.session_state.get("profil", persisted_profile)
    st.sidebar.markdown("## Atlas Menü")
    st.sidebar.markdown(f"**Aktif Profil:** {profile_name}")
    current_currency = get_display_currency()
    chosen_currency = st.sidebar.selectbox("Görüntüleme para birimi", ["USD", "TRY"], index=0 if current_currency == "USD" else 1)
    if chosen_currency != current_currency:
        set_display_currency(chosen_currency)
        st.rerun()
    current_tax = get_estimated_tax_rate()
    chosen_tax = st.sidebar.number_input("Tahmini vergi oranı (%)", min_value=0.0, max_value=100.0, value=float(current_tax), step=1.0)
    if float(chosen_tax) != float(current_tax):
        set_estimated_tax_rate(float(chosen_tax))
        st.rerun()
    st.sidebar.caption("Vergi alanı tahmini hesap içindir.")
    return profile_name, True


def render_sidebar_navigation() -> str:
    pages = ["🏠 Dashboard", "👜 Portföy", "🔎 Radar", "➕ Yeni İşlem", "📜 Geçmiş"]
    current = st.session_state.get("active_page", pages[0])
    page = st.sidebar.radio("Sayfalar", pages, index=pages.index(current) if current in pages else 0)
    st.session_state["active_page"] = page
    return page


def render_portfolio_overview(open_rows: List[Dict[str, Any]], snapshot: Dict[str, float]) -> None:
    risk_summary = get_portfolio_risk_summary(open_rows, snapshot["current_equity"])
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Portföy Durumu</div><div class="section-sub">Sadece rakam değil, risk ve portföy sağlığı da burada.</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card"><div class="metric-label">Net Sermaye</div><div class="metric-value">{format_price(snapshot['net_capital'])}</div></div>
            <div class="metric-card"><div class="metric-label">Toplam Varlık</div><div class="metric-value">{format_price(snapshot['current_equity'])}</div></div>
            <div class="metric-card"><div class="metric-label">Nakit</div><div class="metric-value">{format_price(snapshot['cash'])}</div></div>
            <div class="metric-card"><div class="metric-label">Sistem Getirisi</div><div class="metric-value">%{round(snapshot['total_return_pct'],2)}</div></div>
        </div>
        <div class="metric-grid" style="margin-top:10px;">
            <div class="metric-card"><div class="metric-label">Açık K/Z</div><div class="metric-value">{format_price(snapshot['unrealized_pnl'])}</div></div>
            <div class="metric-card"><div class="metric-label">Tahmini Vergi</div><div class="metric-value">{format_try(snapshot['estimated_tax_try'])}</div></div>
            <div class="metric-card"><div class="metric-label">En Büyük Pozisyon</div><div class="metric-value">%{round(risk_summary['largest_weight']*100,2)}</div></div>
            <div class="metric-card"><div class="metric-label">Çeşitlendirme Skoru</div><div class="metric-value">%{round(risk_summary['diversification_score'],1)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_open_positions(rows: List[Dict[str, Any]], current_equity: float) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Açık Pozisyonlar</div><div class="section-sub">Net PnL, portföy payı ve direkt aksiyonlar.</div>', unsafe_allow_html=True)
    if not rows:
        st.info("Henüz açık pozisyon yok.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    for row in rows:
        row["% Portföy"] = f"%{round((row['Pozisyon Değeri Sayısal'] / max(current_equity, 1e-9)) * 100, 2)}"
    show_df = pd.DataFrame(rows)[["Sembol", "Kaynak", "Alış", "Alım Tutarı", "Güncel", "Net PnL", "PnL %", "% Portföy", "Durum", "Stop", "Hedef", "Not"]]
    st.dataframe(show_df, use_container_width=True, hide_index=True)
    st.markdown("#### Pozisyon kapat")
    for row in rows[:10]:
        with st.expander(f"{row['Sembol']} satış kaydı"):
            c1, c2 = st.columns(2)
            with c1:
                exit_price = st.number_input(f"Satış fiyatı · {row['Sembol']}", min_value=0.0, value=0.0, step=0.01, key=f"exit_{row['ID']}")
            with c2:
                sell_fee = st.number_input(f"Satış komisyonu · {row['Sembol']}", min_value=0.0, value=float(DEFAULT_SELL_FEE), step=0.1, key=f"sellfee_{row['ID']}")
            if st.button(f"{row['Sembol']} kapat", key=f"close_{row['ID']}", use_container_width=True):
                if exit_price > 0:
                    close_portfolio_position(int(row["ID"]), float(exit_price), float(sell_fee))
                    st.success(f"{row['Sembol']} kapatıldı.")
                    st.rerun()
                else:
                    st.warning("Geçerli fiyat gir.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_action_center(profile_name: str, open_rows: List[Dict[str, Any]], snapshot: Dict[str, float]) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Atlas V2 Karar Merkezi</div><div class="section-sub">Yeni fikirleri risk bütçesi ve korelasyon filtresiyle verir.</div>', unsafe_allow_html=True)
    ideas = build_trade_ideas(profile_name, open_rows, snapshot)
    st.write(f"Rejim: **{snapshot['market_regime']}** · Yeni fırsatlar için düşünülebilir alan: **{format_price(snapshot['riskable'])}**")
    if ideas.empty:
        st.info("Şu an filtrelerden geçen yeni fikir yok.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    show = ideas[["Sembol", "Aksiyon", "Atlas Skoru", "Önerilen Tutar", "Önerilen Adet", "Risk Bütçesi", "Tahmini Korelasyon", "Risk/Getiri", "Mesaj"]]
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_radar_page(profile_name: str, open_rows: List[Dict[str, Any]], snapshot: Dict[str, float]) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Radar</div><div class="section-sub">Skor, hacim, trend ve rejim filtresiyle adaylar.</div>', unsafe_allow_html=True)
    radar = build_radar(profile_name, "1G", 12)
    if radar.empty:
        st.info("Radar şu an boş.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    st.dataframe(radar, use_container_width=True, hide_index=True)
    ideas = build_trade_ideas(profile_name, open_rows, snapshot)
    if not ideas.empty:
        st.markdown("#### En uygulanabilir ilk 3 fikir")
        st.dataframe(ideas.head(3), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_add_position(default_symbol: str = "AAPL") -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Yeni İşlem Ekle</div><div class="section-sub">Midas veya manuel aldığın işlemi buraya işle.</div>', unsafe_allow_html=True)
    symbol = st.selectbox("Sembol", SEARCHABLE_ASSETS, index=SEARCHABLE_ASSETS.index(default_symbol) if default_symbol in SEARCHABLE_ASSETS else 0)
    live = get_latest_symbol_price(symbol)
    c1, c2, c3 = st.columns(3)
    with c1:
        entry_price = st.number_input("Alış fiyatı", min_value=0.0, value=float(round(live, 4)) if live else 0.0, step=0.01)
    with c2:
        quantity = st.number_input("Adet", min_value=0.0, value=0.0, step=0.0001, format="%.4f")
    with c3:
        buy_fee = st.number_input("Alış komisyonu", min_value=0.0, value=float(DEFAULT_BUY_FEE), step=0.1)
    source = st.text_input("Kaynak", value="KULLANICI")
    note = st.text_input("Not", value="")
    target_weight_pct = st.number_input("Hedef portföy payı (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    if st.button("Pozisyonu ekle", use_container_width=True):
        if entry_price > 0 and quantity > 0:
            add_portfolio_position(
                symbol=symbol,
                entry_price=float(entry_price),
                quantity=float(quantity),
                note=note,
                source=source,
                target_weight=float(target_weight_pct) / 100.0,
                buy_fee=float(buy_fee),
                gross_amount=float(entry_price) * float(quantity),
            )
            st.success(f"{symbol} eklendi.")
            st.rerun()
        else:
            st.warning("Fiyat ve adet sıfırdan büyük olmalı.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_history_tab() -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Geçmiş</div><div class="section-sub">Kapanan işlemler, nakit hareketleri ve temettüler.</div>', unsafe_allow_html=True)
    hist = load_history(200)
    st.markdown("#### Kapanan İşlemler")
    if hist.empty:
        st.info("Henüz kapanan işlem yok.")
    else:
        st.dataframe(hist, use_container_width=True, hide_index=True)
    st.markdown("#### Nakit Hareketi Ekle")
    c1, c2, c3 = st.columns(3)
    with c1:
        tx_type = st.selectbox("İşlem", ["deposit", "withdrawal"])
    with c2:
        amount = st.number_input("Tutar", min_value=0.0, value=0.0, step=1.0)
    with c3:
        note = st.text_input("Açıklama", value="")
    if st.button("Nakit hareketi ekle", use_container_width=True):
        if amount > 0:
            add_cash_transaction(tx_type, amount, note)
            st.success("Nakit hareketi kaydedildi.")
            st.rerun()
        else:
            st.warning("Tutar gir.")
    st.markdown("#### Temettü Ekle")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        sym = st.selectbox("Temettü sembolü", SEARCHABLE_ASSETS, key="divsym")
    with d2:
        qty = st.number_input("Adet", min_value=0.0, value=0.0, step=0.0001, format="%.4f", key="divqty")
    with d3:
        aps = st.number_input("Hisse başı", min_value=0.0, value=0.0, step=0.0001, format="%.4f", key="divaps")
    with d4:
        withholding = st.number_input("Stopaj %", min_value=0.0, max_value=100.0, value=15.0, step=1.0, key="divwith")
    if st.button("Temettü kaydet", use_container_width=True):
        if qty > 0 and aps > 0:
            add_dividend_record(sym, qty, aps, withholding, note="Manuel kayıt")
            st.success("Temettü kaydedildi.")
            st.rerun()
        else:
            st.warning("Adet ve hisse başı değer gir.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_dashboard_page(profile_name: str, open_rows: List[Dict[str, Any]], snapshot: Dict[str, float]) -> None:
    render_portfolio_overview(open_rows, snapshot)
    render_action_center(profile_name, open_rows, snapshot)
    if PLOTLY_AVAILABLE and snapshot["net_capital"] > 0:
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=21, freq="D")
        base = np.linspace(max(snapshot["net_capital"] * 0.97, 1.0), snapshot["current_equity"], len(idx))
        st.markdown('<div class="card"><div class="section-title">Eğri</div><div class="section-sub">Basit performans eğrisi.</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx, y=base, mode="lines", name="Atlas V2"))
        fig.update_layout(height=280, template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#111827", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


def main_streamlit() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    inject_css()
    init_db()
    profile_name, ready = onboarding_sidebar()
    if not ready:
        render_topbar(None)
        st.info("Atlas V2 kısa onboarding sonrası açılır.")
        return
    if not has_any_user_data():
        seed_midas_demo_portfolio(force=True)
    if st.sidebar.button("Midas demo portföyünü yeniden kur", use_container_width=True):
        seed_midas_demo_portfolio(force=True)
        st.rerun()
    page = render_sidebar_navigation()
    open_rows, invested_now = build_open_positions()
    snapshot = compute_portfolio_snapshot(open_rows, invested_now)
    render_topbar(snapshot)
    if page == "🏠 Dashboard":
        render_dashboard_page(profile_name, open_rows, snapshot)
    elif page == "👜 Portföy":
        render_portfolio_overview(open_rows, snapshot)
        render_open_positions(open_rows, snapshot["current_equity"])
        render_action_center(profile_name, open_rows, snapshot)
    elif page == "🔎 Radar":
        render_radar_page(profile_name, open_rows, snapshot)
    elif page == "➕ Yeni İşlem":
        render_add_position(DEFAULT_SYMBOLS[0])
    else:
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


# ------------------------------
# Tests
# ------------------------------
class TestAtlasMoneyV2(unittest.TestCase):
    def setUp(self) -> None:
        idx = pd.date_range("2025-01-01", periods=260, freq="D")
        base = np.linspace(100, 160, len(idx))
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

    def test_add_indicators_columns(self):
        out = add_indicators(self.df)
        self.assertIn("EMA200", out.columns)
        self.assertIn("ATR14", out.columns)
        self.assertIn("ADX_PROXY", out.columns)

    def test_build_risk_levels(self):
        out = add_indicators(self.df)
        sl, tp, rr = build_risk_levels(out, "AL")
        self.assertIsNotNone(sl)
        self.assertIsNotNone(tp)
        self.assertGreater(rr, 1)

    def test_format_price(self):
        self.assertIn("1.000,50", format_price(1000.5))

    def test_searchable_assets_contains_safe_assets(self):
        self.assertIn("GLD", SEARCHABLE_ASSETS)
        self.assertIn("SGOV", SEARCHABLE_ASSETS)

    def test_profile_presets(self):
        self.assertIn("Dengeli", PROFILE_PRESETS)

    def test_suggest_position_size(self):
        out = suggest_position_size(
            cash=1000,
            current_equity=1200,
            entry_price=100,
            stop_loss=95,
            risk_per_trade=0.01,
        )
        self.assertGreaterEqual(out["gross_amount"], 0)

    def test_calculate_estimated_tax(self):
        hist = pd.DataFrame({"pnl_try": [1000.0, -200.0, 500.0]})
        out = calculate_estimated_tax(hist)
        self.assertGreaterEqual(out["estimated_tax_try"], 0.0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(argv=[sys.argv[0]])
    elif STREAMLIT_AVAILABLE:
        main_streamlit()
    else:
        print_environment_help()
