
import sqlite3
import time
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
APP_SUBTITLE = "Basit Anlatan Yatırım Aracı"
DB_PATH = "signals.db"
INITIAL_CASH = 0.0

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
            asset_type TEXT DEFAULT 'Hisse',
            coupon_date TEXT DEFAULT '',
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
            asset_type TEXT DEFAULT 'Hisse'
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
    portfolio_cols = [r[1] for r in cur.execute("PRAGMA table_info(portfolio)").fetchall()]
    if "asset_type" not in portfolio_cols:
        cur.execute("ALTER TABLE portfolio ADD COLUMN asset_type TEXT DEFAULT 'Hisse'")
    if "coupon_date" not in portfolio_cols:
        cur.execute("ALTER TABLE portfolio ADD COLUMN coupon_date TEXT DEFAULT ''")
    if "entry_fx" not in portfolio_cols:
        cur.execute("ALTER TABLE portfolio ADD COLUMN entry_fx REAL DEFAULT 0")
    history_cols = [r[1] for r in cur.execute("PRAGMA table_info(trades_history)").fetchall()]
    if "asset_type" not in history_cols:
        cur.execute("ALTER TABLE trades_history ADD COLUMN asset_type TEXT DEFAULT 'Hisse'")
    if "entry_fx" not in history_cols:
        cur.execute("ALTER TABLE trades_history ADD COLUMN entry_fx REAL DEFAULT 0")
    if "exit_fx" not in history_cols:
        cur.execute("ALTER TABLE trades_history ADD COLUMN exit_fx REAL DEFAULT 0")
    if "pnl_try" not in history_cols:
        cur.execute("ALTER TABLE trades_history ADD COLUMN pnl_try REAL DEFAULT 0")
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


def convert_money(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    currency = get_display_currency()
    if currency == "TRY":
        return float(value) * get_usdtry_rate()
    return float(value)


def infer_initial_cash() -> float:
    open_df = load_open_portfolio()
    if not open_df.empty:
        try:
            return float((pd.to_numeric(open_df["entry_price"], errors="coerce").fillna(0) * pd.to_numeric(open_df["quantity"], errors="coerce").fillna(0)).sum())
        except Exception:
            return 0.0
    hist = load_history(500)
    if not hist.empty:
        try:
            return float((pd.to_numeric(hist["entry_price"], errors="coerce").fillna(0) * pd.to_numeric(hist["quantity"], errors="coerce").fillna(0)).max())
        except Exception:
            return 0.0
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
    cash_df = load_cash_transactions()
    if not cash_df.empty:
        return
    if not load_history(1).empty:
        return
    inferred = infer_initial_cash()
    if inferred > 0:
        set_initial_cash(inferred, source="auto")


def add_cash_transaction(tx_type: str, amount: float, note: str = "") -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO cash_transactions (created_at, tx_type, amount, note)
        VALUES (?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
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
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    effective_entry_fx = float(entry_fx) if entry_fx and entry_fx > 0 else float(get_usdtry_rate())
    cur.execute(
        """
        INSERT INTO portfolio (created_at, symbol, entry_price, quantity, note, source, target_weight, asset_type, coupon_date, status, entry_fx)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            symbol,
            float(entry_price),
            float(quantity),
            note,
            source,
            float(target_weight),
            asset_type,
            coupon_date,
            effective_entry_fx,
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
        "SELECT created_at, symbol, entry_price, quantity, note, source, asset_type, COALESCE(entry_fx, 0) FROM portfolio WHERE id = ?",
        (int(position_id),),
    ).fetchone()
    if row is None:
        conn.close()
        return
    opened_at, symbol, entry_price, quantity, note, source, asset_type, entry_fx = row
    pnl = (float(exit_price) - float(entry_price)) * float(quantity)
    pnl_pct = ((float(exit_price) / float(entry_price)) - 1) * 100 if float(entry_price) else 0.0
    effective_entry_fx = float(entry_fx) if float(entry_fx or 0) > 0 else float(get_usdtry_rate())
    exit_fx = float(get_usdtry_rate())
    pnl_try = ((float(exit_price) * exit_fx) - (float(entry_price) * effective_entry_fx)) * float(quantity)
    cur.execute(
        """
        INSERT INTO trades_history (opened_at, closed_at, symbol, entry_price, exit_price, quantity, pnl, pnl_pct, note, source, asset_type, entry_fx, exit_fx, pnl_try)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ),
    )
    cur.execute("UPDATE portfolio SET status='closed' WHERE id = ?", (int(position_id),))
    conn.commit()
    conn.close()


def safe_float(value) -> Optional[float]:
    if pd.isna(value):
        return None
    return float(value)


def format_number(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_price(value: Optional[float], is_money: bool = True) -> str:
    if value is None:
        return "-"
    shown = convert_money(value) if is_money else float(value)
    number = format_number(shown)
    if not is_money:
        return number
    currency = get_display_currency()
    return f"${number}" if currency == "USD" else f"₺{number}"


def format_try(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"₺{format_number(value)}"


def calculate_estimated_tax(history_df: pd.DataFrame) -> Dict[str, float]:
    if history_df.empty:
        return {"taxable_profit_try": 0.0, "estimated_tax_try": 0.0, "tax_rate": get_estimated_tax_rate()}
    if "pnl_try" not in history_df.columns:
        taxable_profit_try = 0.0
    else:
        taxable_profit_try = float(pd.to_numeric(history_df["pnl_try"], errors="coerce").fillna(0).clip(lower=0).sum())
    tax_rate = get_estimated_tax_rate()
    estimated_tax_try = taxable_profit_try * (tax_rate / 100.0)
    return {"taxable_profit_try": taxable_profit_try, "estimated_tax_try": estimated_tax_try, "tax_rate": tax_rate}


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
    try:
        time.sleep(0.10)
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
    except Exception:
        return pd.DataFrame(), None, "Analiz sırasında hata oluştu"


@cache_data(ttl=240, show_spinner=False)
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


@cache_data(ttl=900, show_spinner=False)
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
        .flow-step { color:#E2E8F0; font-size:0.95rem; font-weight:700; padding:10px 12px; border-radius:14px; background:#0F172A; border:1px solid rgba(148,163,184,0.08); margin:8px 0; text-align:center; }
        .flow-arrow { text-align:center; color:#64748B; font-size:1.2rem; margin:2px 0; }
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
            <div style="color:#94A3B8;font-size:0.9rem;">Atlas artık trade ile nakit hareketlerini ayrı izler.</div>
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
    usdtry = get_usdtry_rate()
    st.sidebar.markdown(f"**Net Sermaye:** {format_price(get_initial_cash())}")
    st.sidebar.caption(f"Kur: 1 USD ≈ ₺{format_number(usdtry)}")
    current_tax = get_estimated_tax_rate()
    chosen_tax = st.sidebar.number_input("Tahmini vergi oranı (%)", min_value=0.0, max_value=100.0, value=float(current_tax), step=1.0)
    if float(chosen_tax) != float(current_tax):
        set_estimated_tax_rate(float(chosen_tax))
        st.rerun()
    st.sidebar.caption("Vergi alanı bilgilendirme amaçlıdır. Kesin hesap için mali müşavir kontrolü gerekir.")
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
                alert = "Koruyucu stop altı, satış değerlendir"
            elif pnl_pct >= 4:
                action = "SATIŞ DÜŞÜN"
                alert = "Hedef bölgesine yaklaşıyor"
            elif pnl_pct <= -3:
                action = "AZALT / DİKKAT"
                alert = "Zarar artıyor, dikkat gerekli"
        entry_value = float(row["entry_price"]) * float(row["quantity"])
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
                "PnL": format_price(pnl),
                "PnL %": f"%{round(float(pnl_pct), 2)}",
                "% Portföy": "-",
                "Durum": action,
                "Stop": format_price(stop) if stop is not None else "-",
                "Hedef": format_price(target) if target is not None else "-",
                "Hedef Pay": f"%{round(float(row.get('target_weight', 0))*100, 0)}" if float(row.get("target_weight", 0)) > 0 else "-",
                "Hedef Pay Sayısal": float(row.get("target_weight", 0) or 0),
                "Not": str(row["note"] or ""),
                "Uyarı": alert,
                "PnL Sayısal": pnl,
                "Mevcut Pay Sayısal": 0.0,
            }
        )
    return rows, invested_now


def compute_portfolio_snapshot(open_rows: List[Dict[str, Any]], invested_now: float) -> Dict[str, float]:
    flow = get_cash_flow_summary()
    net_capital = flow["net_capital"]
    unrealized_pnl = float(sum(r["PnL Sayısal"] for r in open_rows)) if open_rows else 0.0
    realized_pnl = 0.0
    history_df = load_history(500)
    if not history_df.empty:
        realized_pnl = float(pd.to_numeric(history_df["pnl"], errors="coerce").fillna(0).sum())
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
    riskable = cash * 0.5
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
        "unrealized_pnl": unrealized_pnl,
        "realized_pnl": realized_pnl,
        "dividend_income": net_dividends,
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "after_tax_total_pnl": after_tax_total_pnl_usd,
        "after_tax_return_pct": after_tax_return_pct,
        "taxable_profit_try": tax_summary["taxable_profit_try"],
        "estimated_tax_try": tax_summary["estimated_tax_try"],
        "estimated_tax_rate": tax_summary["tax_rate"],
        "deposits": flow["deposits"],
        "withdrawals": flow["withdrawals"],
    }


def push_prefill_to_new_trade(symbol: str, weight: str, source: str, note: str) -> None:
    st.session_state["portfolio_prefill_symbol"] = symbol
    st.session_state["portfolio_prefill_weight"] = weight
    st.session_state["portfolio_prefill_source"] = source
    st.session_state["portfolio_prefill_note"] = note
    st.session_state["active_page"] = "➕ Yeni İşlem"


def render_portfolio_overview(open_rows: List[Dict[str, Any]], snapshot: Dict[str, float]) -> None:
    risk = "DÜŞÜK" if len(open_rows) <= 2 else "ORTA" if len(open_rows) <= 4 else "YÜKSEK"
    risk_color = "#10B981" if risk == "DÜŞÜK" else "#F59E0B" if risk == "ORTA" else "#EF4444"

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Portföy Durumu</div><div class="section-sub">Toplam varlık, net sermaye, nakit ve yeni işlem için risk alanı.</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card"><div class="metric-label">Net Sermaye</div><div class="metric-value">{format_price(snapshot['net_capital'])}</div></div>
            <div class="metric-card"><div class="metric-label">Portföy Değeri</div><div class="metric-value">{format_price(snapshot['portfolio_value'])}</div></div>
            <div class="metric-card"><div class="metric-label">Nakit</div><div class="metric-value">{format_price(snapshot['cash'])}</div></div>
            <div class="metric-card"><div class="metric-label">Toplam Varlık</div><div class="metric-value">{format_price(snapshot['current_equity'])}</div></div>
        </div>
        <div class="metric-grid" style="margin-top:12px;">
            <div class="metric-card"><div class="metric-label">Sistem Getirisi</div><div class="metric-value">%{round(float(snapshot['total_return_pct']), 2)}</div></div>
            <div class="metric-card"><div class="metric-label">Açık K/Z</div><div class="metric-value">{format_price(snapshot['unrealized_pnl'])}</div></div>
            <div class="metric-card"><div class="metric-label">Net Temettü</div><div class="metric-value">{format_price(snapshot['dividend_income'])}</div></div>
            <div class="metric-card"><div class="metric-label">Tahmini Vergi Etkisi</div><div class="metric-value">{format_try(snapshot['estimated_tax_try'])}</div></div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:10px; align-items:center;">
            <span style="color:#94A3B8;">Yeni fırsatlar için düşünülebilir alan: <strong>{format_price(snapshot['riskable'])}</strong></span>
            <span class="risk-chip" style="color:{risk_color}; border:1px solid {risk_color};">Portföy Riski: {risk}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if PLOTLY_AVAILABLE:
        hist_points = 14
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=hist_points, freq="D")
        atlas = np.linspace(max(snapshot["net_capital"] * 0.97, 1.0), snapshot["current_equity"], hist_points)
        nasdaq = np.linspace(max(snapshot["net_capital"] * 0.98, 1.0), snapshot["net_capital"] * 1.018, hist_points)
        cash_curve = np.linspace(max(snapshot["net_capital"] * 0.99, 1.0), snapshot["net_capital"] * 1.0035, hist_points)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx, y=atlas, mode="lines", name="Atlas"))
        fig.add_trace(go.Scatter(x=idx, y=nasdaq, mode="lines", name="NASDAQ"))
        fig.add_trace(go.Scatter(x=idx, y=cash_curve, mode="lines", name="PPF / Nakit"))
        fig.update_layout(height=300, template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#111827", margin=dict(l=10, r=10, t=20, b=10), legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_open_positions(rows: List[Dict[str, Any]], current_equity: float) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Açık Pozisyonlar</div><div class="section-sub">Mevcut işlemlerini izle, aksiyon önerisini gör ve gerektiğinde sat.</div>', unsafe_allow_html=True)
    if not rows:
        st.info("Henüz açık pozisyon yok.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    for row in rows:
        row["% Portföy"] = f"%{round((row['Pozisyon Değeri Sayısal'] / max(current_equity, 1e-9)) * 100, 2)}"
    show_df = pd.DataFrame(rows)[["Sembol", "Kaynak", "Alış", "Güncel", "PnL", "PnL %", "% Portföy", "Hedef Pay", "Durum", "Stop", "Hedef", "Not"]]
    st.dataframe(show_df, use_container_width=True, hide_index=True)
    st.markdown("#### Pozisyon Sat")
    for row in rows[:8]:
        with st.expander(f"{row['Sembol']} için satış kaydı oluştur"):
            exit_price = st.number_input(f"Satış fiyatı · {row['Sembol']}", min_value=0.0, value=0.0, step=0.01, key=f"exit_{row['ID']}")
            if st.button(f"{row['Sembol']} sat", key=f"close_{row['ID']}", use_container_width=True):
                if exit_price > 0:
                    close_portfolio_position(int(row["ID"]), float(exit_price))
                    st.success(f"{row['Sembol']} satıldı ve geçmişe işlendi.")
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
    st.markdown(f"**Nakit önerisi:** {format_price(cash)} nakdin var. Bunun {format_price(riskable)} kadarı yeni fırsatlar için değerlendirilebilir.")

    radar = build_radar(profile_name, "1G", 10)
    open_map = {row["Sembol"]: row for row in rows}
    yeni_firsatlar: List[Dict[str, Any]] = []
    artirilabilirler: List[Dict[str, Any]] = []

    if not radar.empty:
        for _, item in radar.iterrows():
            sembol = str(item["Sembol"])
            if sembol not in open_map:
                yeni_firsatlar.append(item.to_dict())
                continue
            mevcut = open_map[sembol]
            hedef_pay = float(mevcut.get("Hedef Pay Sayısal", 0) or 0)
            mevcut_pay = float(mevcut.get("Mevcut Pay Sayısal", 0) or 0)
            skor = float(item["Atlas Skoru"])
            if hedef_pay > 0 and mevcut_pay < max(hedef_pay - 0.03, 0) and skor >= 85:
                z = item.to_dict()
                z["Mevcut Pay"] = f"%{round(mevcut_pay * 100, 2)}"
                z["Hedef Pay"] = f"%{round(hedef_pay * 100, 2)}"
                artirilabilirler.append(z)

    if yeni_firsatlar:
        st.markdown("#### Yeni Atlas Fırsatları")
        for i, row in enumerate(yeni_firsatlar[:3]):
            onerilen_tutar = max(10.0, min(riskable, cash * 0.5 if cash > 0 else 0.0))
            st.markdown(
                f"<div class='radar-card'><div class='radar-title'>{row['Sembol']} · {row['Karar']}</div><div class='radar-meta'>Atlas Skoru: %{row['Atlas Skoru']} · Risk/Getiri: {row['Risk/Getiri']} · Önerilen pozisyon: {format_price(onerilen_tutar)}</div></div>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button(f"{row['Sembol']} yeni işleme aktar", key=f"prepare_{row['Sembol']}_{i}", use_container_width=True):
                    push_prefill_to_new_trade(
                        str(row["Sembol"]),
                        "%15",
                        "ATLAS TRADE",
                        f"Atlas fırsatı · Skor %{row['Atlas Skoru']}"
                    )
                    st.rerun()
            with c2:
                st.caption(f"Stop: {format_price(row['Stop'])} · Hedef: {format_price(row['Hedef'])}")

    if artirilabilirler:
        st.markdown("#### Pozisyon Artırılabilirler")
        for i, row in enumerate(artirilabilirler[:3]):
            st.markdown(
                f"<div class='radar-card'><div class='radar-title'>{row['Sembol']} · Pozisyon artırılabilir</div><div class='radar-meta'>Atlas Skoru: %{row['Atlas Skoru']} · Mevcut Pay: {row['Mevcut Pay']} · Hedef Pay: {row['Hedef Pay']}</div></div>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button(f"{row['Sembol']} artırma formuna git", key=f"boost_{row['Sembol']}_{i}", use_container_width=True):
                    push_prefill_to_new_trade(
                        str(row["Sembol"]),
                        row["Hedef Pay"],
                        "ATLAS TRADE",
                        f"Atlas artırma fırsatı · Skor %{row['Atlas Skoru']}",
                    )
                    st.rerun()
            with c2:
                st.caption(f"Stop: {format_price(row['Stop'])} · Hedef: {format_price(row['Hedef'])}")

    if not yeni_firsatlar and not artirilabilirler:
        st.info("Şu an yeni fırsat veya artırılabilir pozisyon görünmüyor.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_initial_portfolio_builder(profile_name: str) -> None:
    suggestion_key = f"atlas_suggestions_{profile_name}"

    if suggestion_key not in st.session_state:
        st.session_state[suggestion_key] = suggest_initial_portfolio(profile_name)

    base_suggestions = st.session_state[suggestion_key]

    if "atlas_hazir_portfoy" not in st.session_state:
        st.session_state["atlas_hazir_portfoy"] = []

    if "atlas_secili_portfoy" not in st.session_state:
        st.session_state["atlas_secili_portfoy"] = {}

    hazir_portfoy = st.session_state["atlas_hazir_portfoy"]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">Atlas Önerilen Başlangıç Portföyü</div>
        <div class="section-sub">
            İstediğin varlıkları seç, Atlas bunları önerilen oranlarla hazırlasın.
            Midas tarafında alımı yaptıktan sonra buradan portföye işle.
        </div>
        """,
        unsafe_allow_html=True
    )

    show_df = pd.DataFrame([
        {k: v for k, v in item.items() if k in ["Varlık", "Tür", "Pay Yazı", "Not"]}
        for item in base_suggestions
    ]).rename(columns={"Pay Yazı": "Pay"})

    st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="flow-step">Portföy adaylarını seç</div>', unsafe_allow_html=True)
    hazir_semboller = {item["Varlık"] for item in hazir_portfoy}
    selectable_suggestions = [item for item in base_suggestions if item["Varlık"] not in hazir_semboller]

    if selectable_suggestions:
        for idx, item in enumerate(selectable_suggestions):
            sembol = item["Varlık"]
            varsayilan = st.session_state["atlas_secili_portfoy"].get(sembol, False)
            secildi = st.checkbox(
                f"{sembol} · {item['Tür']} · {item['Pay Yazı']}",
                value=varsayilan,
                key=f"atlas_select_{sembol}_{idx}",
            )
            st.session_state["atlas_secili_portfoy"][sembol] = secildi
            st.caption(item["Not"])

        secilenler = [
            item for item in selectable_suggestions
            if st.session_state["atlas_secili_portfoy"].get(item["Varlık"], False)
        ]

        if secilenler:
            st.markdown('<div class="flow-arrow">↓</div>', unsafe_allow_html=True)
            if st.button("Seçilen işlemleri hazırla", use_container_width=True, key="atlas_hazirla_btn"):
                mevcutlar = {x["Varlık"] for x in st.session_state["atlas_hazir_portfoy"]}
                for item in secilenler:
                    if item["Varlık"] in mevcutlar:
                        continue
                    st.session_state["atlas_hazir_portfoy"].append(
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
                st.success("Seçilen varlıklar hazırlandı.")
                st.rerun()

    if hazir_portfoy:
        st.markdown('<div class="flow-arrow">↓</div>', unsafe_allow_html=True)
        st.markdown('<div class="flow-step">Midas tarafında aldıklarını işaretle ve portföye ekle</div>', unsafe_allow_html=True)
        baslangic_bakiye = get_cash_flow_summary()["net_capital"]
        silinecekler = []

        for idx, item in enumerate(hazir_portfoy):
            sembol = item["Varlık"]
            st.markdown(f"**{sembol}** · {item['Tür']} · Önerilen pay {item['Pay Yazı']}")

            key_alindi = f"midas_alindi_{sembol}_{idx}"
            key_alis = f"alis_fiyati_{sembol}_{idx}"
            key_adet = f"adet_{sembol}_{idx}"

            if key_alindi not in st.session_state:
                st.session_state[key_alindi] = bool(item.get("Midas Alındı", False))
            if key_alis not in st.session_state:
                st.session_state[key_alis] = float(item.get("Alış Fiyatı", 0.0))
            if key_adet not in st.session_state:
                mevcut_alis = st.session_state[key_alis]
                varsayilan_tutar = baslangic_bakiye * float(item["Pay"])
                varsayilan_adet = 0.0 if mevcut_alis <= 0 else round(varsayilan_tutar / max(mevcut_alis, 1e-9), 4)
                st.session_state[key_adet] = float(item.get("Adet", 0.0)) if float(item.get("Adet", 0.0)) > 0 else float(varsayilan_adet)

            c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1.2])
            with c1:
                midas_alindi = st.checkbox("Midas'ta aldım", key=key_alindi)
            with c2:
                alis_fiyati = st.number_input("Alış fiyatı", min_value=0.0, step=0.01, key=key_alis)
            with c3:
                adet = st.number_input("Adet", min_value=0.0, step=0.0001, key=key_adet)
            with c4:
                pozisyon_degeri = alis_fiyati * adet
                st.markdown(f"Pozisyon değeri: **{format_price(pozisyon_degeri)}**")

            st.session_state["atlas_hazir_portfoy"][idx]["Midas Alındı"] = midas_alindi
            st.session_state["atlas_hazir_portfoy"][idx]["Alış Fiyatı"] = alis_fiyati
            st.session_state["atlas_hazir_portfoy"][idx]["Adet"] = adet

            c5, c6 = st.columns([1, 1])
            with c5:
                if st.button(f"{sembol} portföye ekle", key=f"tekli_portfoy_ekle_{sembol}_{idx}", use_container_width=True):
                    current_item = st.session_state["atlas_hazir_portfoy"][idx]
                    if not current_item["Midas Alındı"]:
                        st.warning(f"Önce {sembol} için Midas alımını onayla.")
                    elif current_item["Alış Fiyatı"] <= 0 or current_item["Adet"] <= 0:
                        st.warning(f"{sembol} için alış fiyatı ve adet girmen gerekiyor.")
                    else:
                        add_portfolio_position(
                            current_item["Varlık"],
                            current_item["Alış Fiyatı"],
                            current_item["Adet"],
                            f"Atlas önerisi · {current_item['Not']} · Önerilen pay {current_item['Pay Yazı']}",
                            "ATLAS TRADE",
                            float(current_item["Pay"]),
                            current_item["Tür"],
                            "",
                        )
                        silinecekler.append(sembol)
                        st.success(f"{sembol} portföye eklendi.")

            with c6:
                if st.button(f"{sembol} formda aç", key=f"tekli_revize_{sembol}_{idx}", use_container_width=True):
                    push_prefill_to_new_trade(
                        sembol,
                        item["Pay Yazı"],
                        "ATLAS TRADE",
                        f"Atlas önerisi · {item['Not']}",
                    )
                    st.rerun()

            st.markdown("---")

        if hazir_portfoy:
            if st.button("Midas'tan aldığım seçili işlemleri toplu ekle", use_container_width=True, key="toplu_midas_ekle"):
                eklendi = 0
                kalanlar = []
                for item in st.session_state["atlas_hazir_portfoy"]:
                    if item.get("Midas Alındı") and float(item.get("Alış Fiyatı", 0)) > 0 and float(item.get("Adet", 0)) > 0:
                        add_portfolio_position(
                            item["Varlık"],
                            float(item["Alış Fiyatı"]),
                            float(item["Adet"]),
                            f"Atlas önerisi · {item['Not']} · Önerilen pay {item['Pay Yazı']}",
                            "ATLAS TRADE",
                            float(item["Pay"]),
                            item["Tür"],
                            "",
                        )
                        eklendi += 1
                    else:
                        kalanlar.append(item)
                st.session_state["atlas_hazir_portfoy"] = kalanlar
                if eklendi > 0:
                    st.success(f"{eklendi} işlem portföye eklendi.")
                    st.rerun()
                else:
                    st.warning("Toplu ekleme için önce seçimleri işaretleyip fiyat ve adet gir.")

        if silinecekler:
            sil_set = set(silinecekler)
            st.session_state["atlas_hazir_portfoy"] = [x for x in st.session_state["atlas_hazir_portfoy"] if x["Varlık"] not in sil_set]
            for sembol in sil_set:
                for key_prefix in ["midas_alindi", "alis_fiyati", "adet"]:
                    for k in [k for k in list(st.session_state.keys()) if k.startswith(f"{key_prefix}_{sembol}_")]:
                        del st.session_state[k]
                if sembol in st.session_state["atlas_secili_portfoy"]:
                    st.session_state["atlas_secili_portfoy"][sembol] = False
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_position_form(default_symbol: str) -> None:
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
    snapshot = compute_portfolio_snapshot(open_rows, invested_now)

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
        asset_type = st.selectbox("Varlık Türü", ["Hisse", "ETF", "Altın ETF", "Tahvil ETF", "PPF / Nakit Park", "Eurobond"], index=0)
        coupon_date = st.text_input("Kupon Tarihi (Eurobond ise)", value="", placeholder="2026-06-15")
        note = st.text_input("Not", value=st.session_state.get("portfolio_prefill_note", "Midas işlemi"))

        selected_weight_value = float(selected_weight.replace("%", "").replace(",", ".")) / 100.0
        position_value_preview = entry * qty
        equity_after = snapshot["current_equity"]
        pct_after = (position_value_preview / max(equity_after, 1e-9)) * 100 if position_value_preview > 0 else 0.0

        st.markdown(
            f"**Ön İzleme:** Pozisyon değeri {format_price(position_value_preview)} · Mevcut toplam varlık {format_price(snapshot['current_equity'])} · Bu işlemin yaklaşık payı %{round(float(pct_after), 2)} · Hedeflenen pay {selected_weight}"
        )
        if pct_after > 40:
            st.warning("Bu işlem portföyde çok büyük ağırlık oluşturuyor. Oranı veya adedi düşürmek daha sağlıklı olabilir.")

        submitted = st.form_submit_button("Portföye Ekle", use_container_width=True)
        if submitted:
            if symbol and entry > 0 and qty > 0:
                add_portfolio_position(symbol, entry, qty, f"{note} · Önerilen pay {selected_weight}", source, selected_weight_value, asset_type, coupon_date)
                st.session_state["portfolio_prefill_symbol"] = symbol
                st.session_state["portfolio_prefill_weight"] = selected_weight
                st.session_state["portfolio_prefill_source"] = source
                st.session_state["portfolio_prefill_note"] = note
                st.success(f"{symbol} portföye eklendi.")
                st.rerun()
            else:
                st.warning("Varlık, alış fiyatı ve adet bilgisi gerekli.")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Hazır formu temizle", use_container_width=True, key="clear_prefill_form"):
            st.session_state["portfolio_prefill_symbol"] = default_symbol
            st.session_state["portfolio_prefill_weight"] = "%20"
            st.session_state["portfolio_prefill_source"] = "KULLANICI"
            st.session_state["portfolio_prefill_note"] = "Midas işlemi"
            st.rerun()
    with c2:
        st.caption("Atlas fırsatlarından gelen semboller burada otomatik açılır.")


def render_cash_transaction_form() -> None:
    tx_mode = st.radio(
        "Nakit işlemi",
        ["Sermaye Girişi", "Sermaye Çıkışı"],
        horizontal=True,
        key="cash_mode_radio",
    )
    amount = st.number_input("Tutar", min_value=0.0, step=100.0, value=0.0, key="cash_amount_input")
    default_note = "Dışarıdan sermaye eklendi" if tx_mode == "Sermaye Girişi" else "Portföyden para çekildi"
    note = st.text_input("Not", value=default_note, key="cash_note_input")

    if st.button(f"{tx_mode} kaydet", use_container_width=True, key="cash_tx_submit"):
        if amount <= 0:
            st.warning("Geçerli bir tutar gir.")
            return
        tx_type = "deposit" if tx_mode == "Sermaye Girişi" else "withdrawal"
        add_cash_transaction(tx_type, amount, note)
        st.success(f"{tx_mode} kaydedildi: {format_price(amount)}")
        st.rerun()


def render_cash_transactions_table() -> None:
    cash_df = load_cash_transactions()
    if cash_df.empty:
        st.info("Henüz sermaye girişi veya çıkışı kaydı yok.")
        return
    show = cash_df.copy()
    show["tx_type"] = show["tx_type"].map({"deposit": "Sermaye Girişi", "withdrawal": "Sermaye Çıkışı"}).fillna(show["tx_type"])
    show["amount"] = show["amount"].apply(lambda x: format_price(float(x)))
    cols = ["created_at", "tx_type", "amount", "note"]
    st.dataframe(show[cols], use_container_width=True, hide_index=True)


def render_dividend_form(default_symbol: str) -> None:
    open_df = load_open_portfolio()
    open_symbols = sorted(open_df["symbol"].astype(str).unique().tolist()) if not open_df.empty else []
    symbol_options = open_symbols if open_symbols else SEARCHABLE_ASSETS
    default_index = symbol_options.index(default_symbol) if default_symbol in symbol_options else 0
    symbol = st.selectbox("Temettü alınan varlık", symbol_options, index=default_index, key="div_symbol")
    quantity = st.number_input("Temettüye konu adet", min_value=0.0, step=0.0001, value=0.0, key="div_qty")
    amount_per_share = st.number_input("Hisse başı temettü", min_value=0.0, step=0.0001, value=0.0, key="div_per_share")
    withholding_rate = st.number_input("Kesilen stopaj (%)", min_value=0.0, max_value=100.0, step=1.0, value=15.0, key="div_withholding")
    note = st.text_input("Not", value="Temettü tahsilatı", key="div_note")

    gross_amount = quantity * amount_per_share
    withholding_tax = gross_amount * (withholding_rate / 100.0)
    net_amount = gross_amount - withholding_tax
    st.caption(f"Brüt: {format_price(gross_amount)} · Kesinti: {format_price(withholding_tax)} · Net: {format_price(net_amount)}")

    if st.button("Temettü kaydını ekle", use_container_width=True, key="div_submit"):
        if not symbol or quantity <= 0 or amount_per_share <= 0:
            st.warning("Varlık, adet ve hisse başı temettü bilgisi gerekli.")
            return
        add_dividend_record(symbol, quantity, amount_per_share, withholding_rate, note)
        st.success(f"{symbol} için temettü kaydı eklendi.")
        st.rerun()


def render_dividend_table() -> None:
    div_df = load_dividend_history()
    if div_df.empty:
        st.info("Henüz temettü kaydı yok.")
        return
    show = div_df.copy()
    for col in ["gross_amount", "withholding_tax", "net_amount"]:
        show[col] = show[col].apply(lambda x: format_price(float(x)))
    show["net_amount_try"] = show["net_amount_try"].apply(lambda x: format_try(float(x)))
    show["withholding_rate"] = show["withholding_rate"].apply(lambda x: f"%{round(float(x), 2)}")
    cols = ["created_at", "symbol", "quantity", "amount_per_share", "gross_amount", "withholding_rate", "withholding_tax", "net_amount", "net_amount_try", "note"]
    st.dataframe(show[cols], use_container_width=True, hide_index=True)


def render_add_position(default_symbol: str) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Yeni İşlem</div><div class="section-sub">Pozisyon ekle, sermaye girişi yap veya sermaye çıkışı kaydet.</div>', unsafe_allow_html=True)

    tx_mode = st.radio(
        "İşlem türü",
        ["Pozisyon Ekle", "Sermaye Girişi", "Sermaye Çıkışı", "Temettü Kaydı"],
        horizontal=True,
        key="new_tx_mode",
    )

    if tx_mode == "Pozisyon Ekle":
        render_position_form(default_symbol)
    elif tx_mode == "Temettü Kaydı":
        render_dividend_form(default_symbol)
    else:
        render_cash_transaction_form()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sermaye Hareketleri</div><div class="section-sub">Sistem dışından para girişi ve çıkışı burada görünür.</div>', unsafe_allow_html=True)
    render_cash_transactions_table()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Temettü Geçmişi</div><div class="section-sub">Brüt temettü, stopaj ve net geçen tutar burada görünür.</div>', unsafe_allow_html=True)
    render_dividend_table()
    st.markdown('</div>', unsafe_allow_html=True)


def render_performance_cards(open_rows: List[Dict[str, Any]], snapshot: Dict[str, float]) -> None:
    history_df = load_history(200)
    closed_count = len(history_df)
    win_rate = 0.0
    avg_pnl_pct = 0.0
    realized_pnl = 0.0

    if not history_df.empty:
        win_rate = float((history_df["pnl"] > 0).mean() * 100)
        avg_pnl_pct = float(history_df["pnl_pct"].mean())
        realized_pnl = float(history_df["pnl"].sum())

    dividend_income = float(snapshot["dividend_income"])
    unrealized_pnl = float(snapshot["unrealized_pnl"])
    total_pnl = realized_pnl + unrealized_pnl + dividend_income
    total_return_pct = float(snapshot["total_return_pct"])
    confidence_score = 55.0 if closed_count == 0 and open_rows else 50.0
    if closed_count > 0:
        confidence_score = max(0.0, min(100.0, (win_rate * 0.55) + (max(avg_pnl_pct, 0) * 6) + min(closed_count, 20)))

    best_asset = "-"
    best_asset_pnl = None
    if open_rows:
        sorted_rows = sorted(open_rows, key=lambda x: x["PnL Sayısal"], reverse=True)
        best_asset = sorted_rows[0]["Sembol"]
        best_asset_pnl = sorted_rows[0]["PnL Sayısal"]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Atlas Performans Özeti</div><div class="section-sub">Trade sonucu ile sermaye hareketlerini ayrı okuyabilmen için güncellendi.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Sermaye", format_price(snapshot["net_capital"]), f"Giriş {format_price(snapshot['deposits'])} / Çıkış {format_price(snapshot['withdrawals'])}")
    c2.metric("Toplam Varlık", format_price(snapshot["current_equity"]), f"Sistem %{round(float(total_return_pct), 2)}")
    c3.metric("Gerçekleşen K/Z", format_price(realized_pnl), f"{closed_count} kapanan işlem")
    c4.metric("Net Temettü", format_price(dividend_income), "Portföye geçen nakit")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Vergi Sonrası K/Z", format_price(snapshot["after_tax_total_pnl"]), f"Tahmini %{round(float(snapshot['after_tax_return_pct']), 2)}")
    d2.metric("Tahmini Vergi", format_try(snapshot["estimated_tax_try"]), f"Oran %{round(float(snapshot['estimated_tax_rate']), 1)}")
    d3.metric("En İyi Açık Pozisyon", best_asset, format_price(best_asset_pnl) if best_asset_pnl is not None else "-")
    d4.metric("Atlas Güven Skoru", f"{round(float(confidence_score), 0)}/100", "İç metrik")

    st.markdown('</div>', unsafe_allow_html=True)


def render_cashflow_calendar(open_rows: List[Dict[str, Any]]) -> None:
    eurobonds = [r for r in open_rows if r.get("Varlık Türü") == "Eurobond" and str(r.get("Kupon Tarihi", "")).strip()]
    if not eurobonds:
        return
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Nakit Akış Takvimi</div><div class="section-sub">Eurobond kupon tarihleri burada takip edilir.</div>', unsafe_allow_html=True)
    data = [{"Varlık": i["Sembol"], "Kupon Tarihi": i["Kupon Tarihi"], "Not": i["Not"]} for i in eurobonds]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_history_tab() -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Geçmiş İşlemler</div><div class="section-sub">Kapatılmış işlemler, gerçekleşen kar/zarar ve işlem geçmişi.</div>', unsafe_allow_html=True)
    hist = load_history(100)
    if hist.empty:
        st.info("Henüz kapatılmış işlem yok.")
    else:
        tax_summary = calculate_estimated_tax(hist)
        st.info(f"Tahmini vergiye esas kâr: {format_try(tax_summary['taxable_profit_try'])} · Tahmini vergi etkisi: {format_try(tax_summary['estimated_tax_try'])}")
        show = hist.copy()
        for col in ["entry_price", "exit_price", "pnl"]:
            show[col] = show[col].apply(lambda x: format_price(float(x)))
        if "pnl_try" in show.columns:
            show["pnl_try"] = show["pnl_try"].apply(lambda x: format_try(float(x)))
        show["pnl_pct"] = show["pnl_pct"].apply(lambda x: f"%{round(float(x), 2)}")
        cols = [c for c in ["symbol", "source", "asset_type", "entry_price", "exit_price", "pnl", "pnl_try", "pnl_pct", "closed_at"] if c in show.columns]
        st.dataframe(show[cols], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_sidebar_navigation() -> str:
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "👜 Portföy"
    page = st.sidebar.radio(
        "Sayfalar",
        ["👜 Portföy", "➕ Yeni İşlem", "🗃️ Geçmiş"],
        index=["👜 Portföy", "➕ Yeni İşlem", "🗃️ Geçmiş"].index(st.session_state["active_page"]),
    )
    st.session_state["active_page"] = page
    return page


def main_streamlit() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    inject_css()
    init_db()
    render_topbar()
    profile_name, ready = onboarding_sidebar()
    if not ready:
        st.markdown('<div class="card"><div class="section-title">Atlas ile Başlangıç</div><div class="section-sub">3 kısa sorudan sonra sana uygun profil belirlenir. Ardından Atlas sana Türkçe ve sade bir başlangıç portföyü önerir.</div></div>', unsafe_allow_html=True)
        return

    page = render_sidebar_navigation()
    open_rows, invested_now = build_open_positions()
    snapshot = compute_portfolio_snapshot(open_rows, invested_now)

    if page == "👜 Portföy":
        if not open_rows:
            render_initial_portfolio_builder(profile_name)
        render_portfolio_overview(open_rows, snapshot)
        render_open_positions(open_rows, snapshot["current_equity"])
        render_action_center(open_rows, snapshot["cash"], snapshot["riskable"], profile_name)
        render_performance_cards(open_rows, snapshot)
        render_cashflow_calendar(open_rows)
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
        self.assertIn("1.000,50", format_price(1000.5))

    def test_searchable_assets_contains_safe_assets(self):
        self.assertIn("GLD", SEARCHABLE_ASSETS)
        self.assertIn("SGOV", SEARCHABLE_ASSETS)

    def test_profile_presets(self):
        self.assertIn("Dengeli", PROFILE_PRESETS)

    def test_suggest_initial_portfolio_count(self):
        suggestions = suggest_initial_portfolio("Dengeli")
        self.assertEqual(len(suggestions), 5)

    def test_calculate_estimated_tax(self):
        hist = pd.DataFrame({"pnl_try": [1000.0, -200.0, 500.0]})
        out = calculate_estimated_tax(hist)
        self.assertGreaterEqual(out["estimated_tax_try"], 0.0)

    def test_dividend_summary_empty(self):
        out = get_dividend_summary()
        self.assertIn("net_dividends", out)


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
