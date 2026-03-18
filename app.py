import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# --- CONFIG & THEME ---
APP_TITLE = "AURA TERMINAL"
DB_PATH = "signals.db"

def apply_aura_style():
    st.markdown("""
        <style>
        .stApp { background-color: #0d1117; color: #c9d1d9; }
        .stMetric { background-color: #161b22 !important; border: 1px solid #30363d !important; border-radius: 12px; }
        div[data-testid="stExpander"] { background-color: #0d1117; border: 1px solid #30363d; }
        .price-up { color: #3fb950; }
        .price-down { color: #f85149; }
        </style>
    """, unsafe_allow_html=True)

# --- DATA MODELS (Orijinal Yapına Sadık) ---
@dataclass
class Position:
    symbol: str
    entry_price: float
    quantity: float
    entry_date: str
    currency: str = "USD"

# --- CORE ENGINE (BIST & USD ENTEGRASYONU) ---
def get_usd_try_live() -> float:
    """Anlık USD/TRY kurunu çeker."""
    try:
        data = yf.download("USDTRY=X", period="1d", interval="1m", progress=False)
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return 32.50
    except:
        return 32.50

def parse_market_symbol(sym: str) -> Tuple[str, str]:
    """Sembolün BIST mi NASDAQ mı olduğunu anlar."""
    s = sym.upper().strip()
    bist_leads = ["THYAO", "EREGL", "ASELS", "SISE", "KCHOL", "TUPRS", "BIMAS"]
    if s.isdigit() or any(k in s for k in bist_leads):
        if not s.endswith(".IS"): return f"{s}.IS", "TRY"
    if s.endswith(".IS"): return s, "TRY"
    return s, "USD"

# --- DATABASE LAYER ---
class AuraDataGate:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS aura_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, qty REAL, price REAL, date TEXT, curr TEXT
                )
            """)

    def add_pos(self, p: Position):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO aura_positions (symbol, qty, price, date, curr) VALUES (?,?,?,?,?)",
                         (p.symbol, p.quantity, p.entry_price, p.entry_date, p.currency))

    def get_pos(self) -> List[Position]:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql("SELECT * FROM aura_positions", conn)
            return [Position(row['symbol'], row['price'], row['qty'], row['date'], row['curr']) for _, row in df.iterrows()]

# --- VISUAL SHOW: TREEMAP ---
def render_treemap_show(positions: List[Position], usd_rate: float):
    if not positions:
        st.info("Portföy boş, şov başlayamıyor.")
        return

    data = []
    for p in positions:
        try:
            curr_price = yf.Ticker(p.symbol).fast_info['last_price']
        except: curr_price = p.entry_price
        
        # USD Normalizasyonu (Büyüklük karşılaştırması için)
        val_usd = (p.quantity * curr_price / usd_rate) if p.currency == "TRY" else (p.quantity * curr_price)
        pnl = ((curr_price / p.entry_price) - 1) * 100
        
        data.append({"Sembol": p.symbol, "Değer (USD)": val_usd, "Performans (%)": pnl, "Adet": p.quantity})

    df = pd.DataFrame(data)
    fig = px.treemap(df, path=['Sembol'], values='Değer (USD)', color='Performans (%)',
                     color_continuous_scale='RdYlGn', color_continuous_midpoint=0)
    fig.update_layout(template="plotly_dark", margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP ---
def main():
    apply_aura_style()
    gate = AuraDataGate()
    usd_rate = get_usd_try_live()

    st.title("💎 AURA Terminal v2.0")
    st.markdown(f"**BIST & Global Portfolio Command Center** | Kur: `{usd_rate:.4f}`")

    # Sidebar: İşlem Girişi
    with st.sidebar:
        st.header("🛒 Yeni İşlem")
        s_in = st.text_input("Sembol (THYAO, AAPL vb.)")
        q_in = st.number_input("Adet", min_value=0.01)
        p_in = st.number_input("Maliyet", min_value=0.01)
        if st.button("Portföye Enjekte Et"):
            final_s, final_curr = parse_market_symbol(s_in)
            gate.add_pos(Position(final_s, p_in, q_in, datetime.now().isoformat(), final_curr))
            st.success("İşlem Başarılı!")
            st.rerun()

    # Dashboard
    pos_list = gate.get_pos()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🌋 Portföy Isı Haritası")
        render_treemap_show(pos_list, usd_rate)

    with col2:
        st.subheader("💰 Varlık Özeti")
        if pos_list:
            # Basit toplam hesabı
            total_usd = sum([(p.quantity * p.entry_price / usd_rate if p.currency=="TRY" else p.quantity * p.entry_price) for p in pos_list])
            st.metric("Toplam Maliyet (USD)", f"${total_usd:,.2f}")
            st.metric("Toplam Maliyet (TRY)", f"₺{total_usd*usd_rate:,.2f}")

    st.subheader("📝 İşlem Geçmişi")
    if pos_list:
        st.table(pd.DataFrame([p.__dict__ for p in pos_list]))

if __name__ == "__main__":
    main()
