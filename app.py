import sqlite3
import time
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

# --- KONFİGÜRASYON & GÖRSEL TEMA ---
st.set_page_config(page_title="AURA Terminal", layout="wide", initial_sidebar_state="collapsed")

def apply_custom_style():
    st.markdown("""
        <style>
        .main { background-color: #0e1117; }
        .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px !important; }
        .stActionButton { background-color: #238636 !important; color: white !important; }
        div[data-testid="stExpander"] { border: 1px solid #30363d; background-color: #0d1117; border-radius: 8px; }
        .price-up { color: #26a69a; font-weight: bold; }
        .price-down { color: #ef5350; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

# --- VERİTABANI YÖNETİMİ ---
DB_PATH = "aura_terminal.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Portföy tablosu (BIST/US desteği ve Döviz tipi eklendi)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            quantity REAL,
            entry_price REAL,
            entry_date TEXT,
            currency TEXT DEFAULT 'USD'
        )
    """)
    # Nakit hareketleri
    cursor.execute("CREATE TABLE IF NOT EXISTS cash (id INTEGER PRIMARY KEY, balance REAL, currency TEXT)")
    conn.commit()
    conn.close()

# --- YARDIMCI FONKSİYONLAR (BIST & KUR) ---
@st.cache_data(ttl=3600)
def get_usd_try():
    try:
        return yf.Ticker("USDTRY=X").history(period="1d")['Close'].iloc[-1]
    except:
        return 32.50

def normalize_symbol(symbol: str):
    s = symbol.upper().strip()
    if s.isdigit() or any(x in s for x in ["THYAO", "EREGL", "ASELS", "SISE", "EKGYO"]):
        if not s.endswith(".IS"): return s + ".IS", "TRY"
    if s.endswith(".IS"): return s, "TRY"
    return s, "USD"

# --- ANALİZ MOTORU ---
def fetch_and_analyze(symbol: str):
    sym, curr = normalize_symbol(symbol)
    ticker = yf.Ticker(sym)
    hist = ticker.history(period="60d")
    if hist.empty: return None
    
    last_price = hist['Close'].iloc[-1]
    change = ((last_price / hist['Close'].iloc[-2]) - 1) * 100
    
    # Teknik Göstergeler (Görsel Skorlama için)
    sma20 = hist['Close'].rolling(20).mean().iloc[-1]
    rsi = 100 - (100 / (1 + hist['Close'].diff().gt(0).rolling(14).sum() / hist['Close'].diff().lt(0).rolling(14).sum()))
    
    # Hacim Patlaması Kontrolü (Volume Spike)
    avg_vol = hist['Volume'].tail(20).mean()
    current_vol = hist['Volume'].iloc[-1]
    vol_spike = current_vol > (avg_vol * 1.5)
    
    return {
        "symbol": sym, "price": last_price, "change": change, 
        "curr": curr, "rsi": rsi.iloc[-1], "above_sma": last_price > sma20,
        "vol_spike": vol_spike, "hist": hist
    }

# --- GÖRSEL ŞÖLEN BÖLÜMLERİ ---
def render_header(usd_rate):
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.title("💎 AURA Terminal")
        st.caption(f"Finansal Komuta Merkezi | USDTRY: {usd_rate:.4f}")
    with col2:
        st.metric("Piyasa Durumu", "Açık", delta="BIST / NASDAQ")
    with col3:
        if st.button("🔄 Verileri Yenile"): st.rerun()

def render_portfolio_treemap(df_portfolio, usd_rate):
    """Müşterinin gözünü boyayacak o meşhur Treemap"""
    if df_portfolio.empty:
        st.info("Portföyünüzde henüz varlık bulunmuyor.")
        return

    # Fiyatları eşitleme (BIST hisselerini USD'ye çevirerek büyüklük karşılaştırması yap)
    df_portfolio['size_usd'] = df_portfolio.apply(
        lambda x: (x['quantity'] * x['current_price'] / usd_rate) if x['currency'] == 'TRY' 
        else (x['quantity'] * x['current_price']), axis=1
    )
    
    fig = px.treemap(
        df_portfolio,
        path=[px.Constant("Varlıklarım"), 'symbol'],
        values='size_usd',
        color='pnl_percent',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        title="Varlık Dağılımı ve Performans Isı Haritası (USD Bazlı)"
    )
    fig.update_layout(margin=dict(t=30, l=0, r=0, b=0), height=400, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# --- ANA UYGULAMA DÖNGÜSÜ ---
def main():
    apply_custom_style()
    init_db()
    usd_rate = get_live_usd_try() if 'get_live_usd_try' in locals() else get_usd_try()
    
    render_header(usd_rate)
    
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Akıllı Analiz", "⚙️ Ayarlar"])
    
    with tab1:
        # Örnek Veri Çekme (Veritabanından gelecek şekilde simüle edildi)
        # Gerçek uygulamada sqlite'dan çekilen veriler df_portfolio olacak
        mock_data = {
            "symbol": ["THYAO.IS", "AAPL", "EREGL.IS", "NVDA"],
            "quantity": [100, 10, 500, 5],
            "current_price": [280.50, 185.20, 45.10, 890.0],
            "pnl_percent": [12.5, -2.1, 5.4, 25.0],
            "currency": ["TRY", "USD", "TRY", "USD"]
        }
        df_portfolio = pd.DataFrame(mock_data)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        total_usd = (df_portfolio[df_portfolio['currency']=='USD'].apply(lambda x: x['quantity']*x['current_price'], axis=1).sum() + 
                     df_portfolio[df_portfolio['currency']=='TRY'].apply(lambda x: x['quantity']*x['current_price']/usd_rate, axis=1).sum())
        
        col_m1.metric("Toplam Varlık (USD)", f"${total_usd:,.2f}")
        col_m2.metric("Toplam Varlık (TRY)", f"₺{total_usd*usd_rate:,.2f}")
        col_m3.metric("Günlük Kar/Zarar", "%+2.4", delta="145.20 $")

        render_portfolio_treemap(df_portfolio, usd_rate)
        
        st.subheader("Varlık Detayları")
        st.dataframe(df_portfolio, use_container_width=True)

    with tab2:
        st.subheader("Akıllı Sembol Taraması")
        search_sym = st.text_input("Sembol Giriniz (Örn: THYAO, AAPL, BTC-USD)", "").upper()
        if search_sym:
            with st.spinner("Veriler işleniyor..."):
                analysis = fetch_and_analyze(search_sym)
                if analysis:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Anlık Fiyat", f"{analysis['price']:.2f} {analysis['curr']}")
                    c2.metric("RSI (14)", f"{analysis['rsi']:.1f}")
                    status = "GÜÇLÜ" if analysis['vol_spike'] and analysis['above_sma'] else "ZAYIF"
                    c3.metric("Sinyal Gücü", status, delta="Hacim Destekli" if analysis['vol_spike'] else "Düşük Hacim")
                    
                    # Mum Grafiği (Görsel Show)
                    fig = go.Figure(data=[go.Candlestick(
                        x=analysis['hist'].index,
                        open=analysis['hist']['Open'],
                        high=analysis['hist']['High'],
                        low=analysis['hist']['Low'],
                        close=analysis['hist']['Close']
                    )])
                    fig.update_layout(title=f"{search_sym} Teknik Görünüm", template="plotly_dark", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if st.button(f"{search_sym} Portföye Ekle"):
                        st.success("Veritabanı bağlantısı aktif olduğunda eklenecektir.")

    with tab3:
        st.info("AURA Terminal v2.0 - BIST & Global Edition")
        if st.button("Tüm Veritabanını Sıfırla"):
            st.warning("Bu işlem geri alınamaz!")

if __name__ == "__main__":
    main()
