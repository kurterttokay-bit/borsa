import sqlite3
import time
import unittest
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

# --- KONFİGÜRASYON & İSİMLENDİRME ---
APP_TITLE = "AURA TERMINAL v2.0"
DB_PATH = "aura_finance.db"

# --- 1. VERİTABANI VE MODEL KATMANI ---
class AuraDB:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Portföy: Sembol, Adet, Maliyet, Tarih, Para Birimi
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_date TEXT NOT NULL,
                    currency TEXT NOT NULL
                )
            """)
            # Nakit Akışı
            cursor.execute("CREATE TABLE IF NOT EXISTS cash_flow (id INTEGER PRIMARY KEY, amount REAL, currency TEXT)")
            conn.commit()

    def add_position(self, symbol: str, qty: float, price: float, currency: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO positions (symbol, quantity, entry_price, entry_date, currency) VALUES (?, ?, ?, ?, ?)",
                (symbol.upper(), qty, price, datetime.now().isoformat(), currency)
            )

    def get_all_positions(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM positions", conn)

# --- 2. TEKNİK ANALİZ MOTORU ---
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 20: return df
    df = df.copy()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # EMA
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    # Volatilite (ATR)
    high_low = df['High'] - df['Low']
    df['ATR'] = high_low.rolling(window=14).mean()
    return df

# --- 3. AKILLI BIST VE KUR SİSTEMİ ---
@st.cache_data(ttl=3600)
def get_usd_try_rate():
    try:
        data = yf.download("USDTRY=X", period="1d", progress=False)
        return data['Close'].iloc[-1]
    except:
        return 32.50

def get_clean_symbol(symbol: str) -> Tuple[str, str]:
    s = symbol.upper().strip()
    # Türk hissesi tespiti (Sadece rakam veya bilinen BIST kodları)
    bist_keywords = ["THYAO", "EREGL", "ASELS", "SISE", "KCHOL", "TUPRS"]
    if s.isdigit() or any(k in s for k in bist_keywords):
        if not s.endswith(".IS"): return s + ".IS", "TRY"
    if s.endswith(".IS"): return s, "TRY"
    return s, "USD"

# --- 4. GÖRSEL ŞÖLEN KOMPONENTLERİ ---
def apply_ui_style():
    st.markdown("""
        <style>
        .stApp { background-color: #0b0e14; color: #e0e0e0; }
        .metric-card {
            background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
            border: 1px solid #30363d;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        .stMetric { border: 1px solid #30363d; padding: 10px; border-radius: 10px; background: #161b22; }
        </style>
    """, unsafe_allow_html=True)

def plot_fancy_treemap(df_p, usd_rate):
    if df_p.empty: return
    # Tüm değerleri USD'ye eşitle (Görsel boyut için)
    df_p['val_usd'] = df_p.apply(lambda x: (x['qty'] * x['price'] / usd_rate) if x['curr'] == 'TRY' else (x['qty'] * x['price']), axis=1)
    
    fig = px.treemap(
        df_p, path=[px.Constant("Portföy"), 'symbol'], values='val_usd',
        color='pnl_pct', color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        title="Varlık Dağılımı ve Performans Isı Haritası"
    )
    fig.update_layout(template="plotly_dark", margin=dict(t=30, l=0, r=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# --- 5. ANA EKRAN (STREAMLIT) ---
def main():
    apply_ui_style()
    db = AuraDB()
    usd_rate = get_usd_try_rate()

    st.title(f"💎 {APP_TITLE}")
    st.caption(f"Veri Odaklı Yatırım Terminali | Canlı USD/TRY: {usd_rate:.4f}")

    # --- SIDEBAR: YENİ İŞLEM ---
    with st.sidebar:
        st.header("➕ Yeni Pozisyon")
        with st.form("trade_form"):
            sym_input = st.text_input("Sembol (Örn: THYAO veya AAPL)")
            qty_input = st.number_input("Adet", min_value=0.1)
            price_input = st.number_input("Giriş Fiyatı", min_value=0.01)
            submitted = st.form_submit_button("Portföye Ekle")
            
            if submitted and sym_input:
                clean_s, curr = get_clean_symbol(sym_input)
                db.add_position(clean_s, qty_input, price_input, curr)
                st.success(f"{clean_s} Eklendi!")
                st.rerun()

    # --- DASHBOARD ---
    df_db = db.get_all_positions()
    
    tab1, tab2 = st.tabs(["📊 Portföy Analizi", "🎯 Akıllı Radar"])

    with tab1:
        if not df_db.empty:
            # Anlık Fiyatları Çek (Hızlı Erişim)
            symbols = df_db['symbol'].unique().tolist()
            prices = {}
            for s in symbols:
                try:
                    t = yf.Ticker(s)
                    prices[s] = t.fast_info['last_price']
                except: prices[s] = 0

            # Tabloyu Hazırla
            df_display = df_db.copy()
            df_display['current'] = df_display['symbol'].map(prices)
            df_display['pnl_pct'] = (df_display['current'] / df_display['entry_price'] - 1) * 100
            
            # Üst Metrikler
            c1, c2, c3 = st.columns(3)
            total_val_usd = sum([
                (row['quantity'] * row['current'] / usd_rate) if row['currency'] == 'TRY' 
                else (row['quantity'] * row['current']) 
                for _, row in df_display.iterrows()
            ])
            c1.metric("Toplam Değer (USD)", f"${total_val_usd:,.2f}")
            c2.metric("Toplam Değer (TRY)", f"₺{total_val_usd*usd_rate:,.2f}")
            c3.metric("Piyasa Durumu", "Aktif", delta="Global")

            # Treemap "Şovu"
            plot_fancy_treemap(
                pd.DataFrame({
                    'symbol': df_display['symbol'],
                    'qty': df_display['quantity'],
                    'price': df_display['current'],
                    'pnl_pct': df_display['pnl_pct'],
                    'curr': df_display['currency']
                }), usd_rate
            )

            st.subheader("📜 Aktif Pozisyonlar")
            st.dataframe(df_display[['symbol', 'quantity', 'entry_price', 'current', 'pnl_pct', 'currency']], use_container_width=True)
        else:
            st.info("Henüz pozisyon açılmamış. Yan menüden ilk hisseni ekle!")

    with tab2:
        st.subheader("🔍 Teknik Analiz Radarı")
        radar_sym = st.text_input("Analiz Edilecek Sembol", value="THYAO")
        if radar_sym:
            s_final, _ = get_clean_symbol(radar_sym)
            data = yf.download(s_final, period="60d", progress=False)
            if not data.empty:
                data = add_indicators(data)
                
                # Candlestick Chart
                fig = go.Figure(data=[go.Candlestick(
                    x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
                    name=s_final
                )])
                # Göstergeleri Ekle
                fig.add_trace(go.Scatter(x=data.index, y=data['EMA20'], name='EMA20', line=dict(color='orange', width=1)))
                fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Sinyal Durumu
                last_rsi = data['RSI'].iloc[-1]
                col_r1, col_r2 = st.columns(2)
                col_r1.write(f"**RSI (14):** {last_rsi:.2f}")
                if last_rsi < 30: col_r1.success("Aşırı Satım (Fırsat?)")
                elif last_rsi > 70: col_r1.error("Aşırı Alım (Dikkat!)")
                else: col_r1.info("Nötr Bölge")

if __name__ == "__main__":
    main()
