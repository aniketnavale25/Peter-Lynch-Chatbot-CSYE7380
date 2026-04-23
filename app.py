"""
app.py  --  Peter Lynch ChatBot  --  Streamlit UI
Tab 1: Chat with Peter Lynch      (RAG-powered)
Tab 2: Lynch Stock Analyzer       (single stock deep-dive + backtest)
Tab 3: Financial Ratios Dashboard (fin_data_df, Dow Jones 30)
Tab 4: K-Means Stock Screener     (professor's exact algorithm)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from lynch_rag import ask_lynch, load_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Northeastern University brand colors
#   Primary red : #C8102E
#   Black       : #000000
#   Dark bg     : #1A1A1A
#   Card bg     : #2A2A2A
#   Border      : #3D3D3D
#   Muted text  : #A0A0A0
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Peter Lynch ChatBot",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #1A1A1A;
    color: #E8E8E8;
}
[data-testid="stSidebar"] {
    background: #000000;
    border-right: 2px solid #C8102E;
}
h1, h2, h3, h4 {
    font-family: 'Lato', sans-serif;
    font-weight: 900;
    color: #FFFFFF;
}
.chat-user {
    background: #2A2A2A;
    border-left: 3px solid #C8102E;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.95rem;
}
.chat-lynch {
    background: #222222;
    border-left: 3px solid #FFFFFF;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.95rem;
    line-height: 1.65;
}
.speaker-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.user-label  { color: #C8102E; }
.lynch-label { color: #FFFFFF; }
.metric-card {
    background: #2A2A2A;
    border: 1px solid #3D3D3D;
    border-top: 3px solid #C8102E;
    border-radius: 6px;
    padding: 16px 18px;
    text-align: center;
}
.metric-card .value {
    font-size: 1.85rem;
    font-weight: 900;
    color: #C8102E;
}
.metric-card .label {
    font-size: 0.72rem;
    color: #A0A0A0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}
.verdict-undervalued { color: #4CAF50; font-weight: 700; font-size:1.05rem; }
.verdict-fair        { color: #FFC107; font-weight: 700; font-size:1.05rem; }
.verdict-overvalued  { color: #C8102E; font-weight: 700; font-size:1.05rem; }
.stButton > button {
    background: #C8102E !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover { background: #A00D24 !important; }
.stTextInput > div > div > input {
    background: #2A2A2A !important;
    color: #E8E8E8 !important;
    border: 1px solid #3D3D3D !important;
    border-radius: 6px !important;
}
.stTabs [role="tab"] {
    font-weight: 700;
    font-size: 0.85rem;
    color: #A0A0A0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stTabs [aria-selected="true"] {
    color: #C8102E !important;
    border-bottom: 2px solid #C8102E !important;
    background: transparent !important;
}
hr { border-color: #3D3D3D; }
.scorecard-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 10px;
    margin: 3px 0;
    background: #2A2A2A;
    border-radius: 6px;
    font-size: 0.85rem;
}
.long-card {
    background: #1A2A1A; border: 1px solid #4CAF50;
    border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;
}
.short-card {
    background: #2A1A1A; border: 1px solid #C8102E;
    border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────────────────────────────────────────
# Shared static financial data (mirrors yahooquery financial_data fields)
# ─────────────────────────────────────────────────────────────────────────────
FIN_DATA_STATIC = {
    "AAPL": {"currentPrice":202.5,"returnOnEquity":1.365,"returnOnAssets":0.225,"debtToEquity":145.0,"currentRatio":0.92,"profitMargins":0.243,"earningsGrowth":0.101,"revenueGrowth":0.040,"grossMargins":0.465,"ebitdaMargins":0.347,"operatingMargins":0.345,"totalCash":5.38e10,"totalDebt":9.68e10,"totalRevenue":3.96e11,"grossProfits":1.84e11,"freeCashflow":9.38e10,"operatingCashflow":1.08e11,"revenuePerShare":25.97,"quickRatio":0.78},
    "AMGN": {"currentPrice":315.0,"returnOnEquity":0.676,"returnOnAssets":0.064,"debtToEquity":1035.9,"currentRatio":1.26,"profitMargins":0.122,"earningsGrowth":-0.187,"revenueGrowth":0.109,"grossMargins":0.687,"ebitdaMargins":0.453,"operatingMargins":0.518,"totalCash":1.20e10,"totalDebt":6.09e10,"totalRevenue":3.34e10,"grossProfits":2.30e10,"freeCashflow":1.37e10,"operatingCashflow":1.15e10,"revenuePerShare":62.24,"quickRatio":0.84},
    "AMZN": {"currentPrice":195.0,"returnOnEquity":0.243,"returnOnAssets":0.074,"debtToEquity":54.3,"currentRatio":1.06,"profitMargins":0.093,"earningsGrowth":0.846,"revenueGrowth":0.105,"grossMargins":0.489,"ebitdaMargins":0.189,"operatingMargins":0.113,"totalCash":1.01e11,"totalDebt":1.55e11,"totalRevenue":6.38e11,"grossProfits":3.12e11,"freeCashflow":4.46e10,"operatingCashflow":1.16e11,"revenuePerShare":60.92,"quickRatio":0.84},
    "AXP":  {"currentPrice":271.0,"returnOnEquity":0.347,"returnOnAssets":0.038,"debtToEquity":183.3,"currentRatio":1.35,"profitMargins":0.167,"earningsGrowth":0.157,"revenueGrowth":0.106,"grossMargins":0.648,"ebitdaMargins":0.0,"operatingMargins":0.178,"totalCash":4.11e10,"totalDebt":5.55e10,"totalRevenue":6.08e10,"grossProfits":3.94e10,"freeCashflow":0.0,"operatingCashflow":1.40e10,"revenuePerShare":85.34,"quickRatio":1.34},
    "BA":   {"currentPrice":163.0,"returnOnEquity":0.0,"returnOnAssets":-0.043,"debtToEquity":0.0,"currentRatio":1.32,"profitMargins":-0.178,"earningsGrowth":0.0,"revenueGrowth":-0.308,"grossMargins":-0.026,"ebitdaMargins":-0.123,"operatingMargins":-0.239,"totalCash":2.63e10,"totalDebt":5.60e10,"totalRevenue":6.65e10,"grossProfits":-1.74e9,"freeCashflow":-9.69e9,"operatingCashflow":-1.21e10,"revenuePerShare":102.8,"quickRatio":0.39},
    "CAT":  {"currentPrice":285.0,"returnOnEquity":0.553,"returnOnAssets":0.097,"debtToEquity":200.1,"currentRatio":1.42,"profitMargins":0.167,"earningsGrowth":0.096,"revenueGrowth":-0.05,"grossMargins":0.325,"ebitdaMargins":0.241,"operatingMargins":0.192,"totalCash":6.17e9,"totalDebt":3.90e10,"totalRevenue":6.48e10,"grossProfits":2.11e10,"freeCashflow":5.22e9,"operatingCashflow":1.20e10,"revenuePerShare":133.2,"quickRatio":0.81},
    "CRM":  {"currentPrice":279.0,"returnOnEquity":0.103,"returnOnAssets":0.047,"debtToEquity":19.7,"currentRatio":1.06,"profitMargins":0.164,"earningsGrowth":0.195,"revenueGrowth":0.076,"grossMargins":0.772,"ebitdaMargins":0.294,"operatingMargins":0.212,"totalCash":1.40e10,"totalDebt":1.21e10,"totalRevenue":3.79e10,"grossProfits":2.93e10,"freeCashflow":1.42e10,"operatingCashflow":1.31e10,"revenuePerShare":39.4,"quickRatio":0.93},
    "CSCO": {"currentPrice":60.6,"returnOnEquity":0.200,"returnOnAssets":0.066,"debtToEquity":71.2,"currentRatio":0.87,"profitMargins":0.170,"earningsGrowth":-0.061,"revenueGrowth":0.094,"grossMargins":0.651,"ebitdaMargins":0.269,"operatingMargins":0.223,"totalCash":1.76e10,"totalDebt":3.24e10,"totalRevenue":5.42e10,"grossProfits":3.53e10,"freeCashflow":1.33e10,"operatingCashflow":1.36e10,"revenuePerShare":13.52,"quickRatio":0.66},
    "CVX":  {"currentPrice":164.8,"returnOnEquity":0.113,"returnOnAssets":0.056,"debtToEquity":19.3,"currentRatio":1.06,"profitMargins":0.090,"earningsGrowth":0.515,"revenueGrowth":0.086,"grossMargins":0.395,"ebitdaMargins":0.205,"operatingMargins":0.114,"totalCash":6.79e9,"totalDebt":2.96e10,"totalRevenue":1.96e11,"grossProfits":7.73e10,"freeCashflow":1.63e10,"operatingCashflow":3.15e10,"revenuePerShare":108.0,"quickRatio":0.71},
    "DIS":  {"currentPrice":112.0,"returnOnEquity":0.059,"returnOnAssets":0.043,"debtToEquity":42.4,"currentRatio":0.68,"profitMargins":0.061,"earningsGrowth":0.346,"revenueGrowth":0.048,"grossMargins":0.367,"ebitdaMargins":0.199,"operatingMargins":0.168,"totalCash":5.49e9,"totalDebt":4.53e10,"totalRevenue":9.25e10,"grossProfits":3.40e10,"freeCashflow":1.08e10,"operatingCashflow":1.50e10,"revenuePerShare":50.83,"quickRatio":0.55},
    "GS":   {"currentPrice":580.0,"returnOnEquity":0.118,"returnOnAssets":0.010,"debtToEquity":575.0,"currentRatio":0.0,"profitMargins":0.280,"earningsGrowth":0.088,"revenueGrowth":0.099,"grossMargins":0.517,"ebitdaMargins":0.0,"operatingMargins":0.318,"totalCash":2.41e11,"totalDebt":3.67e11,"totalRevenue":5.36e10,"grossProfits":2.77e10,"freeCashflow":0.0,"operatingCashflow":4.20e10,"revenuePerShare":163.0,"quickRatio":0.0},
    "HD":   {"currentPrice":350.0,"returnOnEquity":0.0,"returnOnAssets":0.191,"debtToEquity":0.0,"currentRatio":1.24,"profitMargins":0.103,"earningsGrowth":0.021,"revenueGrowth":0.045,"grossMargins":0.331,"ebitdaMargins":0.162,"operatingMargins":0.143,"totalCash":3.45e9,"totalDebt":5.14e10,"totalRevenue":1.53e11,"grossProfits":5.06e10,"freeCashflow":1.75e10,"operatingCashflow":2.34e10,"revenuePerShare":152.7,"quickRatio":0.27},
    "HON":  {"currentPrice":215.0,"returnOnEquity":0.303,"returnOnAssets":0.083,"debtToEquity":178.6,"currentRatio":1.30,"profitMargins":0.147,"earningsGrowth":0.070,"revenueGrowth":0.050,"grossMargins":0.383,"ebitdaMargins":0.197,"operatingMargins":0.175,"totalCash":9.49e9,"totalDebt":2.05e10,"totalRevenue":3.71e10,"grossProfits":1.42e10,"freeCashflow":5.64e9,"operatingCashflow":7.62e9,"revenuePerShare":54.93,"quickRatio":0.88},
    "IBM":  {"currentPrice":240.0,"returnOnEquity":0.220,"returnOnAssets":0.045,"debtToEquity":253.9,"currentRatio":1.04,"profitMargins":0.117,"earningsGrowth":0.648,"revenueGrowth":0.013,"grossMargins":0.536,"ebitdaMargins":0.213,"operatingMargins":0.126,"totalCash":1.47e10,"totalDebt":5.92e10,"totalRevenue":6.28e10,"grossProfits":3.37e10,"freeCashflow":1.07e10,"operatingCashflow":1.37e10,"revenuePerShare":68.06,"quickRatio":0.97},
    "JNJ":  {"currentPrice":152.0,"returnOnEquity":0.219,"returnOnAssets":0.088,"debtToEquity":42.5,"currentRatio":1.14,"profitMargins":0.200,"earningsGrowth":0.080,"revenueGrowth":0.040,"grossMargins":0.645,"ebitdaMargins":0.298,"operatingMargins":0.247,"totalCash":1.89e10,"totalDebt":3.64e10,"totalRevenue":8.88e10,"grossProfits":5.73e10,"freeCashflow":1.52e10,"operatingCashflow":2.00e10,"revenuePerShare":37.20,"quickRatio":0.97},
    "JPM":  {"currentPrice":240.0,"returnOnEquity":0.163,"returnOnAssets":0.013,"debtToEquity":182.0,"currentRatio":0.0,"profitMargins":0.321,"earningsGrowth":0.180,"revenueGrowth":0.133,"grossMargins":0.541,"ebitdaMargins":0.0,"operatingMargins":0.345,"totalCash":5.60e11,"totalDebt":5.62e11,"totalRevenue":1.80e11,"grossProfits":9.74e10,"freeCashflow":0.0,"operatingCashflow":6.80e10,"revenuePerShare":58.94,"quickRatio":0.0},
    "KO":   {"currentPrice":68.0,"returnOnEquity":0.400,"returnOnAssets":0.100,"debtToEquity":196.3,"currentRatio":1.06,"profitMargins":0.230,"earningsGrowth":0.060,"revenueGrowth":0.030,"grossMargins":0.598,"ebitdaMargins":0.299,"operatingMargins":0.249,"totalCash":9.37e9,"totalDebt":3.57e10,"totalRevenue":4.72e10,"grossProfits":2.82e10,"freeCashflow":9.52e9,"operatingCashflow":1.16e10,"revenuePerShare":10.93,"quickRatio":0.97},
    "MCD":  {"currentPrice":295.0,"returnOnEquity":0.0,"returnOnAssets":0.145,"debtToEquity":0.0,"currentRatio":1.50,"profitMargins":0.321,"earningsGrowth":0.018,"revenueGrowth":0.026,"grossMargins":0.599,"ebitdaMargins":0.482,"operatingMargins":0.440,"totalCash":5.31e9,"totalDebt":3.78e10,"totalRevenue":2.55e10,"grossProfits":1.53e10,"freeCashflow":7.48e9,"operatingCashflow":9.70e9,"revenuePerShare":34.50,"quickRatio":1.40},
    "MMM":  {"currentPrice":138.0,"returnOnEquity":0.360,"returnOnAssets":0.090,"debtToEquity":63.6,"currentRatio":1.42,"profitMargins":0.258,"earningsGrowth":5.000,"revenueGrowth":0.016,"grossMargins":0.460,"ebitdaMargins":0.218,"operatingMargins":0.187,"totalCash":3.40e9,"totalDebt":1.32e10,"totalRevenue":2.38e10,"grossProfits":1.10e10,"freeCashflow":3.90e9,"operatingCashflow":5.50e9,"revenuePerShare":41.40,"quickRatio":0.88},
    "MRK":  {"currentPrice":88.0,"returnOnEquity":0.410,"returnOnAssets":0.098,"debtToEquity":76.6,"currentRatio":1.22,"profitMargins":0.268,"earningsGrowth":-0.280,"revenueGrowth":0.070,"grossMargins":0.744,"ebitdaMargins":0.350,"operatingMargins":0.280,"totalCash":1.02e10,"totalDebt":3.58e10,"totalRevenue":6.34e10,"grossProfits":4.72e10,"freeCashflow":1.37e10,"operatingCashflow":1.81e10,"revenuePerShare":25.10,"quickRatio":0.91},
    "MSFT": {"currentPrice":400.5,"returnOnEquity":0.343,"returnOnAssets":0.146,"debtToEquity":34.0,"currentRatio":1.35,"profitMargins":0.354,"earningsGrowth":0.102,"revenueGrowth":0.123,"grossMargins":0.694,"ebitdaMargins":0.543,"operatingMargins":0.455,"totalCash":7.16e10,"totalDebt":1.03e11,"totalRevenue":2.62e11,"grossProfits":1.82e11,"freeCashflow":5.20e10,"operatingCashflow":1.26e11,"revenuePerShare":35.22,"quickRatio":1.20},
    "NKE":  {"currentPrice":58.0,"returnOnEquity":0.347,"returnOnAssets":0.098,"debtToEquity":86.0,"currentRatio":2.22,"profitMargins":0.100,"earningsGrowth":-0.243,"revenueGrowth":-0.077,"grossMargins":0.447,"ebitdaMargins":0.138,"operatingMargins":0.112,"totalCash":9.76e9,"totalDebt":1.21e10,"totalRevenue":4.90e10,"grossProfits":2.19e10,"freeCashflow":5.20e9,"operatingCashflow":6.12e9,"revenuePerShare":32.62,"quickRatio":1.34},
    "NVDA": {"currentPrice":108.5,"returnOnEquity":1.192,"returnOnAssets":0.574,"debtToEquity":12.9,"currentRatio":4.44,"profitMargins":0.558,"earningsGrowth":0.836,"revenueGrowth":0.779,"grossMargins":0.750,"ebitdaMargins":0.638,"operatingMargins":0.611,"totalCash":4.32e10,"totalDebt":1.03e10,"totalRevenue":1.30e11,"grossProfits":9.79e10,"freeCashflow":4.42e10,"operatingCashflow":6.41e10,"revenuePerShare":5.31,"quickRatio":3.67},
    "PG":   {"currentPrice":163.0,"returnOnEquity":0.311,"returnOnAssets":0.109,"debtToEquity":67.4,"currentRatio":0.76,"profitMargins":0.184,"earningsGrowth":0.341,"revenueGrowth":0.021,"grossMargins":0.517,"ebitdaMargins":0.286,"operatingMargins":0.272,"totalCash":1.02e10,"totalDebt":3.47e10,"totalRevenue":8.43e10,"grossProfits":4.36e10,"freeCashflow":1.14e10,"operatingCashflow":1.90e10,"revenuePerShare":35.78,"quickRatio":0.49},
    "SHW":  {"currentPrice":336.0,"returnOnEquity":0.690,"returnOnAssets":0.101,"debtToEquity":298.7,"currentRatio":0.79,"profitMargins":0.116,"earningsGrowth":0.357,"revenueGrowth":0.009,"grossMargins":0.485,"ebitdaMargins":0.190,"operatingMargins":0.127,"totalCash":2.10e8,"totalDebt":1.21e10,"totalRevenue":2.31e10,"grossProfits":1.12e10,"freeCashflow":1.92e9,"operatingCashflow":3.15e9,"revenuePerShare":92.03,"quickRatio":0.39},
    "TRV":  {"currentPrice":260.0,"returnOnEquity":0.189,"returnOnAssets":0.032,"debtToEquity":29.8,"currentRatio":0.32,"profitMargins":0.108,"earningsGrowth":0.284,"revenueGrowth":0.099,"grossMargins":0.267,"ebitdaMargins":0.157,"operatingMargins":0.224,"totalCash":5.46e9,"totalDebt":8.31e9,"totalRevenue":4.64e10,"grossProfits":1.24e10,"freeCashflow":1.65e10,"operatingCashflow":9.07e9,"revenuePerShare":203.6,"quickRatio":0.19},
    "UNH":  {"currentPrice":511.0,"returnOnEquity":0.151,"returnOnAssets":0.071,"debtToEquity":79.7,"currentRatio":0.83,"profitMargins":0.036,"earningsGrowth":0.026,"revenueGrowth":0.068,"grossMargins":0.223,"ebitdaMargins":0.087,"operatingMargins":0.077,"totalCash":2.91e10,"totalDebt":8.18e10,"totalRevenue":4.00e11,"grossProfits":8.94e10,"freeCashflow":1.16e10,"operatingCashflow":2.42e10,"revenuePerShare":434.6,"quickRatio":0.75},
    "V":    {"currentPrice":339.5,"returnOnEquity":0.512,"returnOnAssets":0.167,"debtToEquity":53.8,"currentRatio":1.12,"profitMargins":0.543,"earningsGrowth":0.078,"revenueGrowth":0.101,"grossMargins":0.978,"ebitdaMargins":0.695,"operatingMargins":0.687,"totalCash":1.43e10,"totalDebt":2.06e10,"totalRevenue":3.68e10,"grossProfits":3.60e10,"freeCashflow":1.39e10,"operatingCashflow":2.17e10,"revenuePerShare":19.66,"quickRatio":0.71},
    "VZ":   {"currentPrice":40.5,"returnOnEquity":0.185,"returnOnAssets":0.050,"debtToEquity":170.7,"currentRatio":0.63,"profitMargins":0.130,"earningsGrowth":0.0,"revenueGrowth":0.016,"grossMargins":0.600,"ebitdaMargins":0.362,"operatingMargins":0.212,"totalCash":4.19e9,"totalDebt":1.72e11,"totalRevenue":1.35e11,"grossProfits":8.09e10,"freeCashflow":1.60e10,"operatingCashflow":3.69e10,"revenuePerShare":31.96,"quickRatio":0.47},
    "WMT":  {"currentPrice":85.8,"returnOnEquity":0.214,"returnOnAssets":0.071,"debtToEquity":63.6,"currentRatio":0.82,"profitMargins":0.029,"earningsGrowth":-0.023,"revenueGrowth":0.041,"grossMargins":0.249,"ebitdaMargins":0.062,"operatingMargins":0.042,"totalCash":9.04e9,"totalDebt":6.21e10,"totalRevenue":6.81e11,"grossProfits":1.69e11,"freeCashflow":7.93e9,"operatingCashflow":3.64e10,"revenuePerShare":84.69,"quickRatio":0.20},
}
DOW30 = list(FIN_DATA_STATIC.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='font-size:1.5rem;margin-bottom:0;color:#FFFFFF;'>Peter Lynch</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#A0A0A0;font-size:0.8rem;margin-top:4px;'>"
                "Fidelity Magellan Fund </p>",
                unsafe_allow_html=True)
    st.markdown("<div style='height:2px;background:#C8102E;margin:10px 0;'></div>",
                unsafe_allow_html=True)

    st.markdown("<p style='color:#FFFFFF;font-weight:700;margin-bottom:6px;'>Core Principles</p>",
                unsafe_allow_html=True)
    for p in ["Invest in what you know", "PEG < 1 = Bargain",
              "Avoid the whisper stock", "Know why you own a stock",
              "Patience beats market-timing", "Boring companies can be great"]:
        st.markdown(
            f"<div style='font-size:0.78rem;padding:3px 0 3px 8px;"
            f"border-left:2px solid #C8102E;color:#A0A0A0;margin:2px 0;'>{p}</div>",
            unsafe_allow_html=True)

    st.markdown("<div style='height:2px;background:#3D3D3D;margin:10px 0;'></div>",
                unsafe_allow_html=True)

    if not st.session_state.pipeline_ready:
        if st.button("Load Lynch Knowledge Base"):
            with st.spinner("Indexing documents..."):
                try:
                    load_pipeline()
                    st.session_state.pipeline_ready = True
                    st.success("Ready!")
                except Exception as exc:
                    st.error(f"Error: {exc}")
    else:
        st.success("Knowledge base loaded")
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("<div style='height:2px;background:#3D3D3D;margin:10px 0;'></div>",
                unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem;color:#555555;'>"
                "Powered by LangChain · ChromaDB · Groq<br>"
                "Northeastern University · CSYE 7380</p>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_analyzer, tab_dashboard, tab_kmeans = st.tabs([
    "Chat with Peter Lynch",
    "Lynch Stock Analyzer",
    "Financial Dashboard",
    "K-Means Stock Screener",
])


# =============================================================================
# TAB 1 -- CHAT
# =============================================================================
with tab_chat:
    st.markdown(
        "<h2 style='margin-bottom:0.2rem;'>Ask Peter Lynch</h2>"
        "<p style='color:#A0A0A0;font-size:0.9rem;'>"
        "Ask about investment strategy, stock selection, portfolio management, or his career.</p>",
        unsafe_allow_html=True)
    st.divider()

    with st.container():
        if not st.session_state.chat_history:
            st.markdown(
                "<div style='color:#555555;font-style:italic;text-align:center;"
                "padding:3rem 0;'>Start the conversation below</div>",
                unsafe_allow_html=True)
        else:
            for turn in st.session_state.chat_history:
                st.markdown(
                    f"<div class='speaker-label user-label'>You</div>"
                    f"<div class='chat-user'>{turn['user']}</div>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"<div class='speaker-label lynch-label'>Peter Lynch</div>"
                    f"<div class='chat-lynch'>{turn['assistant']}</div>",
                    unsafe_allow_html=True)

    st.divider()
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_question = st.text_input(
            label="question", placeholder="e.g. How do you use the PEG ratio?",
            label_visibility="collapsed", key="chat_input")
    with col_btn:
        send = st.button("Ask", width='stretch')

    st.markdown("<p style='color:#555555;font-size:0.8rem;margin-top:8px;'>Try asking:</p>",
                unsafe_allow_html=True)
    suggestions = ["What is the PEG ratio?", "How do you identify a ten-bagger?",
                   "What sectors do you prefer?", "How do you size a position?"]
    for col, sug in zip(st.columns(len(suggestions)), suggestions):
        with col:
            if st.button(sug, key=f"sug_{sug}", width='stretch'):
                user_question = sug
                send = True

    if send and user_question.strip():
        if not st.session_state.pipeline_ready:
            st.warning("Please load the knowledge base first (sidebar).")
        else:
            with st.spinner("Peter Lynch is thinking..."):
                answer = ask_lynch(question=user_question, history=st.session_state.chat_history)
            st.session_state.chat_history.append({"user": user_question, "assistant": answer})
            st.rerun()


# =============================================================================
# TAB 2 -- LYNCH STOCK ANALYZER
# Single-stock deep dive: fundamentals, scorecard, backtest
# Uses yfinance for live data
# =============================================================================
with tab_analyzer:
    st.markdown(
        "<h2 style='margin-bottom:0.2rem;'>Lynch Stock Analyzer</h2>"
        "<p style='color:#A0A0A0;font-size:0.9rem;'>"
        "Evaluate any stock through Peter Lynch's PEG-ratio lens and backtest his strategy.</p>",
        unsafe_allow_html=True)
    st.divider()

    col_t, col_b = st.columns([4, 1])
    with col_t:
        ticker_input = st.text_input(
            "ticker", placeholder="e.g. META, AAPL, MSFT",
            label_visibility="collapsed")
    with col_b:
        analyze_btn = st.button("Analyze", width='stretch')

    # ── Helpers ───────────────────────────────────────────────────────────────
    def safe_get(info, *keys, default=None):
        for k in keys:
            v = info.get(k)
            if v is not None:
                return v
        return default

    def fetch_fundamentals(ticker):
        try:
            from yahooquery import Ticker
            obj = Ticker(ticker)
            info = obj.financial_data.get(ticker, {})
            summary = obj.summary_detail.get(ticker, {})
            profile = obj.asset_profile.get(ticker, {})
            key_stats = obj.key_stats.get(ticker, {})

            if not isinstance(info, dict):
                info = {}
            if not isinstance(summary, dict):
                summary = {}
            if not isinstance(profile, dict):
                profile = {}
            if not isinstance(key_stats, dict):
                key_stats = {}

            all_data = {**info, **summary, **profile, **key_stats}

            pe         = all_data.get("trailingPE") or all_data.get("forwardPE")
            eps_growth = all_data.get("earningsGrowth") or all_data.get("revenueGrowth")
            if eps_growth is not None:
                eps_growth *= 100
            peg = None
            if pe and eps_growth and eps_growth > 0:
                peg = pe / eps_growth

            market_cap = all_data.get("marketCap")
            if market_cap:
                if market_cap >= 1e12:   mc_str = f"${market_cap/1e12:.1f}T"
                elif market_cap >= 1e9:  mc_str = f"${market_cap/1e9:.1f}B"
                else:                    mc_str = f"${market_cap/1e6:.0f}M"
            else:
                mc_str = "N/A"

            price      = all_data.get("currentPrice") or all_data.get("regularMarketPrice")
            shares     = key_stats.get("sharesOutstanding")
            cash_total = all_data.get("totalCash")
            revenue_g  = all_data.get("revenueGrowth")
            if revenue_g is not None:
                revenue_g *= 100
            inst_own    = key_stats.get("heldPercentInstitutions")
            insider_own = key_stats.get("heldPercentInsiders")
            dividend    = summary.get("dividendYield") or summary.get("trailingAnnualDividendYield")

            def classify():
                if eps_growth is None:     return "Unknown", "#A0A0A0"
                if eps_growth < 5:         return "Slow Grower", "#A0A0A0"
                if eps_growth < 15:        return "Stalwart", "#4A90D9"
                if eps_growth >= 20:       return "Fast Grower", "#4CAF50"
                return "Moderate Grower", "#FFC107"

            category, cat_color = classify()
            return {
                "name":           profile.get("longName") or ticker,
                "sector":         profile.get("sector", "N/A"),
                "industry":       profile.get("industry", "N/A"),
                "price":          price,
                "pe":             pe,
                "eps_growth":     eps_growth,
                "revenue_growth": revenue_g,
                "peg":            peg,
                "market_cap":     mc_str,
                "debt_equity":    all_data.get("debtToEquity"),
                "roe":            all_data.get("returnOnEquity"),
                "dividend":       dividend,
                "cash_per_share": (cash_total/shares) if cash_total and shares else None,
                "fcf":            all_data.get("freeCashflow"),
                "ps_ratio":       all_data.get("priceToSalesTrailing12Months") or summary.get("priceToSalesTrailing12Months"),
                "current_ratio":  all_data.get("currentRatio"),
                "inst_own":       inst_own,
                "insider_own":    insider_own,
                "category":       category,
                "cat_color":      cat_color,
                "description":    profile.get("longBusinessSummary", ""),
            }
        except Exception as exc:
            return {"error": str(exc), "name": ticker}

    def fetch_history(ticker, period="5y"):
        try:
            from yahooquery import Ticker
            obj = Ticker(ticker)
            df = obj.history(period=period)
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker, level="symbol")
            df.index = pd.to_datetime(df.index)
            df.columns = [c.capitalize() for c in df.columns]
            return df
        except Exception:
            return pd.DataFrame()

    def lynch_backtest(hist):
        if hist.empty or len(hist) < 250:
            return pd.DataFrame()
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        df = pd.DataFrame(index=hist.index)
        df["Close"] = hist["Close"].squeeze().astype(float)
        df["MA200"] = df["Close"].rolling(200).mean()
        df["MA50"]  = df["Close"].rolling(50).mean()
        df["signal"] = 0
        in_trade = False
        for i in range(200, len(df)):
            close = float(df["Close"].iloc[i])
            ma200 = float(df["MA200"].iloc[i])
            if not in_trade and close > ma200 and close < ma200 * 1.10:
                df.iloc[i, df.columns.get_loc("signal")] = 1
                in_trade = True
            elif in_trade and close > ma200 * 1.30:
                df.iloc[i, df.columns.get_loc("signal")] = -1
                in_trade = False
        df["daily_ret"] = df["Close"].pct_change()
        df["position"]  = df["signal"].replace(0, np.nan).ffill().fillna(0).clip(lower=0)
        df["strat_ret"] = df["position"] * df["daily_ret"]
        df["strat_cum"] = (1 + df["strat_ret"]).cumprod()
        df["bnh_cum"]   = (1 + df["daily_ret"]).cumprod()
        return df.dropna(subset=["MA200"])

    # ── Output ────────────────────────────────────────────────────────────────
    if analyze_btn and ticker_input.strip():
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        for ticker in tickers:
            with st.spinner(f"Fetching data for {ticker}..."):
                fund = fetch_fundamentals(ticker)
                hist = fetch_history(ticker)

            if "error" in fund:
                st.error(f"{ticker}: {fund['error']}")
                continue

            # Company header
            st.markdown(
                f"<h3 style='margin-top:1.5rem;color:#FFFFFF;'>{fund['name']} "
                f"<span style='color:#A0A0A0;font-size:0.95rem;'>({ticker})</span></h3>"
                f"<p style='color:#A0A0A0;font-size:0.85rem;margin-top:-8px;'>"
                f"{fund['sector']} · {fund['industry']}</p>",
                unsafe_allow_html=True)

            # Category badge
            st.markdown(
                f"<span style='background:{fund['cat_color']}22;color:{fund['cat_color']};"
                f"border:1px solid {fund['cat_color']}55;border-radius:6px;"
                f"padding:3px 10px;font-size:0.8rem;font-weight:700;'>"
                f"Lynch Category: {fund['category']}</span>",
                unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Metric row 1
            price_str = f"${fund['price']:.2f}"    if fund["price"]         else "N/A"
            pe_str    = f"{fund['pe']:.1f}x"        if fund["pe"]            else "N/A"
            epsg_str  = f"{fund['eps_growth']:.1f}%" if fund["eps_growth"] is not None else "N/A"
            roe_str   = f"{fund['roe']*100:.1f}%"   if fund["roe"] is not None else "N/A"
            div_str   = f"{fund['dividend']*100:.2f}%" if fund["dividend"]   else "0%"
            for col, val, lbl in zip(st.columns(5),
                [price_str, pe_str, epsg_str, roe_str, div_str],
                ["Price", "P/E Ratio", "EPS Growth", "ROE", "Dividend Yield"]):
                col.markdown(
                    f"<div class='metric-card'><div class='value'>{val}</div>"
                    f"<div class='label'>{lbl}</div></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Metric row 2
            peg_str = f"{fund['peg']:.2f}"             if fund["peg"]            else "N/A"
            de_str  = f"{fund['debt_equity']:.0f}%"    if fund["debt_equity"]    else "N/A"
            ps_str  = f"{fund['ps_ratio']:.1f}x"       if fund["ps_ratio"]       else "N/A"
            cr_str  = f"{fund['current_ratio']:.1f}x"  if fund["current_ratio"]  else "N/A"
            ins_str = f"{fund['insider_own']*100:.1f}%" if fund["insider_own"]    else "N/A"
            for col, val, lbl in zip(st.columns(5),
                [peg_str, de_str, ps_str, cr_str, ins_str],
                ["PEG Ratio", "Debt/Equity", "P/S Ratio", "Current Ratio", "Insider Ownership"]):
                col.markdown(
                    f"<div class='metric-card'><div class='value'>{val}</div>"
                    f"<div class='label'>{lbl}</div></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Scorecard + description
            score_col, desc_col = st.columns([3, 2])
            with score_col:
                st.markdown("**Lynch Full Scorecard**")

                def check(label, value, good, warn, bad, fmt=None, invert=False):
                    if value is None:
                        icon, color, display = "o", "#555555", "N/A"
                    else:
                        display = fmt(value) if fmt else str(value)
                        if invert:
                            if value <= good:   icon, color = "v", "#4CAF50"
                            elif value <= warn: icon, color = "!", "#FFC107"
                            else:               icon, color = "x", "#C8102E"
                        else:
                            if value >= good:   icon, color = "v", "#4CAF50"
                            elif value >= warn: icon, color = "!", "#FFC107"
                            else:               icon, color = "x", "#C8102E"
                    icon_html = {"v": "&#10003;", "!": "&#9888;", "x": "&#10007;", "o": "&#9711;"}
                    st.markdown(
                        f"<div class='scorecard-row'>"
                        f"<span style='color:#C0C8D0;'>"
                        f"<span style='color:{color};font-weight:700;'>{icon_html[icon]}</span>"
                        f" {label}</span>"
                        f"<span style='color:{color};font-weight:700;'>{display}</span>"
                        f"</div>", unsafe_allow_html=True)

                check("PEG Ratio (< 1 = bargain)", fund["peg"],
                      good=1.0, warn=2.0, bad=99, fmt=lambda v: f"{v:.2f}", invert=True)
                check("EPS Growth (> 15% = fast grower)", fund["eps_growth"],
                      good=15, warn=5, bad=0, fmt=lambda v: f"{v:.1f}%")
                check("Debt/Equity (< 80% = healthy)", fund["debt_equity"],
                      good=40, warn=80, bad=999, fmt=lambda v: f"{v:.0f}%", invert=True)
                check("ROE (> 15% = strong)",
                      fund["roe"]*100 if fund["roe"] else None,
                      good=15, warn=10, bad=0, fmt=lambda v: f"{v:.1f}%")
                check("Current Ratio (> 2 = safe)", fund["current_ratio"],
                      good=2.0, warn=1.0, bad=0, fmt=lambda v: f"{v:.1f}x")
                check("Insider Ownership (> 5% = skin in game)",
                      fund["insider_own"]*100 if fund["insider_own"] else None,
                      good=5, warn=1, bad=0, fmt=lambda v: f"{v:.1f}%")
                check("Institutional Ownership (< 50% = undiscovered)",
                      fund["inst_own"]*100 if fund["inst_own"] else None,
                      good=50, warn=75, bad=999, fmt=lambda v: f"{v:.1f}%", invert=True)
                check("P/S Ratio (< 2 = reasonable)", fund["ps_ratio"],
                      good=1.0, warn=2.0, bad=99, fmt=lambda v: f"{v:.1f}x", invert=True)

                scores = sum([
                    1 if fund["peg"] and fund["peg"] < 1 else 0,
                    1 if fund["eps_growth"] and fund["eps_growth"] > 15 else 0,
                    1 if fund["debt_equity"] and fund["debt_equity"] < 80 else 0,
                    1 if fund["roe"] and fund["roe"] > 0.15 else 0,
                    1 if fund["current_ratio"] and fund["current_ratio"] > 2 else 0,
                    1 if fund["insider_own"] and fund["insider_own"] > 0.05 else 0,
                ])
                if scores >= 5:    verdict, vc = "Strong Lynch Pick",          "#4CAF50"
                elif scores >= 3:  verdict, vc = "Decent Candidate",           "#FFC107"
                else:              verdict, vc = "Doesn't Pass Lynch's Test",  "#C8102E"
                st.markdown(
                    f"<div style='margin-top:12px;padding:10px 14px;"
                    f"background:{vc}22;border:1px solid {vc}55;border-radius:8px;"
                    f"font-weight:700;color:{vc};font-size:0.95rem;'>"
                    f"{verdict} &nbsp;·&nbsp; {scores}/6 criteria met</div>",
                    unsafe_allow_html=True)

            with desc_col:
                if fund["description"]:
                    st.markdown("**About the Company**")
                    st.markdown(
                        f"<div style='font-size:0.83rem;color:#A0A0A0;line-height:1.6;'>"
                        f"{fund['description'][:500]}...</div>",
                        unsafe_allow_html=True)

            # Charts
            if not hist.empty:
                bt = lynch_backtest(hist)
                chart_col, bt_col = st.columns([3, 2])

                with chart_col:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist.index.tolist(),
                        y=hist["Close"].squeeze().astype(float).tolist(),
                        mode="lines", name="Price",
                        line=dict(color="#4A90D9", width=1.5)))
                    if not bt.empty:
                        fig.add_trace(go.Scatter(
                            x=bt.index.tolist(), y=bt["MA200"].tolist(),
                            mode="lines", name="200-day MA",
                            line=dict(color="#C8102E", width=1, dash="dot")))
                        fig.add_trace(go.Scatter(
                            x=bt.index.tolist(), y=bt["MA50"].tolist(),
                            mode="lines", name="50-day MA",
                            line=dict(color="#FFC107", width=1, dash="dot")))
                        buys  = bt[bt["signal"] ==  1]
                        sells = bt[bt["signal"] == -1]
                        if not buys.empty:
                            fig.add_trace(go.Scatter(
                                x=buys.index.tolist(), y=buys["Close"].tolist(),
                                mode="markers", name="Buy Signal",
                                marker=dict(color="#4CAF50", symbol="triangle-up", size=10)))
                        if not sells.empty:
                            fig.add_trace(go.Scatter(
                                x=sells.index.tolist(), y=sells["Close"].tolist(),
                                mode="markers", name="Sell Signal",
                                marker=dict(color="#C8102E", symbol="triangle-down", size=10)))
                    fig.update_layout(
                        title=f"{ticker} -- 5-Year Price + Moving Averages",
                        paper_bgcolor="#1A1A1A", plot_bgcolor="#2A2A2A",
                        font=dict(color="#E8E8E8", family="Lato"),
                        legend=dict(bgcolor="#2A2A2A", bordercolor="#3D3D3D", font=dict(size=11)),
                        xaxis=dict(gridcolor="#3D3D3D"), yaxis=dict(gridcolor="#3D3D3D", title="Price ($)"),
                        margin=dict(l=10, r=10, t=40, b=10), height=320)
                    st.plotly_chart(fig, width='stretch')

                with bt_col:
                    if not bt.empty:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(
                            x=bt.index.tolist(), y=bt["strat_cum"].tolist(),
                            mode="lines", name="Lynch Strategy",
                            line=dict(color="#C8102E", width=2)))
                        fig2.add_trace(go.Scatter(
                            x=bt.index.tolist(), y=bt["bnh_cum"].tolist(),
                            mode="lines", name="Buy & Hold",
                            line=dict(color="#4A90D9", width=1.5, dash="dot")))
                        fig2.update_layout(
                            title="Strategy vs Buy & Hold",
                            paper_bgcolor="#1A1A1A", plot_bgcolor="#2A2A2A",
                            font=dict(color="#E8E8E8", family="Lato"),
                            legend=dict(bgcolor="#2A2A2A", bordercolor="#3D3D3D", font=dict(size=11)),
                            xaxis=dict(gridcolor="#3D3D3D"),
                            yaxis=dict(gridcolor="#3D3D3D", title="Cumulative Return"),
                            margin=dict(l=10, r=10, t=40, b=10), height=320)
                        st.plotly_chart(fig2, width='stretch')

                        strat_final = bt["strat_cum"].iloc[-1]
                        bnh_final   = bt["bnh_cum"].iloc[-1]
                        n_trades    = int((bt["signal"] != 0).sum())
                        st.markdown(
                            f"<div style='font-size:0.82rem;color:#A0A0A0;text-align:center;'>"
                            f"Strategy: <b style='color:#C8102E;'>+{(strat_final-1)*100:.1f}%</b>"
                            f" &nbsp;|&nbsp; "
                            f"Buy & Hold: <b style='color:#4A90D9;'>+{(bnh_final-1)*100:.1f}%</b>"
                            f" &nbsp;|&nbsp; Signals fired: <b>{n_trades}</b></div>",
                            unsafe_allow_html=True)
            st.divider()

    elif analyze_btn:
        st.warning("Please enter at least one ticker symbol.")

    with st.expander("Lynch's PEG Ratio Guide"):
        st.markdown("""
**PEG Ratio** = P/E / Earnings Growth Rate

| PEG Value | Lynch's Interpretation |
|-----------|----------------------|
| < 0.5     | Hidden gem -- potentially a ten-bagger |
| 0.5 - 1.0 | Bargain -- good risk/reward |
| 1.0 - 2.0 | Fairly priced -- acceptable if story holds |
| > 2.0     | Expensive -- growth must be exceptional |
        """)


# =============================================================================
# TAB 3 -- FINANCIAL RATIOS DASHBOARD
# fin_data_df: key ratios for all selected stocks (Dow Jones 30 default)
# =============================================================================
with tab_dashboard:
    st.markdown(
        "<h2 style='margin-bottom:0.2rem;'>Financial Ratios Dashboard</h2>"
        "<p style='color:#A0A0A0;font-size:0.9rem;'>"
        "Key financial ratios for all selected stocks (fin_data_df). "
        "Default portfolio: Dow Jones 30.</p>",
        unsafe_allow_html=True)
    st.divider()

    if "overview_tickers" not in st.session_state:
        st.session_state.overview_tickers = list(DOW30)

    ov_c1, ov_c2, ov_c3 = st.columns([2, 1, 2])
    with ov_c1:
        new_t = st.text_input("Add", placeholder="e.g. TSLA",
                              label_visibility="collapsed", key="ov_add_input")
    with ov_c2:
        if st.button("Add Stock", key="ov_add_btn") and new_t.strip():
            t = new_t.strip().upper()
            if t not in st.session_state.overview_tickers:
                st.session_state.overview_tickers.append(t)
                st.rerun()
    with ov_c3:
        rm = st.selectbox("Remove", ["--"] + st.session_state.overview_tickers, key="ov_remove")
        if rm != "--":
            st.session_state.overview_tickers.remove(rm)
            st.rerun()

    def build_fin_display(tickers):
        rows = []
        missing = [t for t in tickers if t not in FIN_DATA_STATIC]
        live_data = {}
        if missing:
            try:
                from yahooquery import Ticker
                obj = Ticker(missing)
                fin = obj.financial_data
                summary = obj.summary_detail
                for t in missing:
                    d = {}
                    if isinstance(fin.get(t), dict):
                        d.update(fin[t])
                    if isinstance(summary.get(t), dict):
                        d.update(summary[t])
                    if d:
                        live_data[t] = d
            except Exception:
                pass

        for t in tickers:
            if t in FIN_DATA_STATIC:
                d = FIN_DATA_STATIC[t]
            elif t in live_data:
                d = live_data[t]
            else:
                d = {}

            eg  = d.get("earningsGrowth")
            rg  = d.get("revenueGrowth")
            roe = d.get("returnOnEquity")
            roa = d.get("returnOnAssets")
            rows.append({
                "Symbol":            t,
                "Price ($)":         d.get("currentPrice"),
                "ROE (%)":           round(roe*100,1) if roe is not None else None,
                "ROA (%)":           round(roa*100,1) if roa is not None else None,
                "D/E":               d.get("debtToEquity"),
                "Current Ratio":     d.get("currentRatio"),
                "Profit Margin (%)": round(d.get("profitMargins",0)*100,1) if d.get("profitMargins") is not None else None,
                "EPS Growth (%)":    round(eg*100,1) if eg is not None else None,
                "Rev Growth (%)":    round(rg*100,1) if rg is not None else None,
                "Gross Margin (%)":  round(d.get("grossMargins",0)*100,1) if d.get("grossMargins") is not None else None,
                "Op Margin (%)":     round(d.get("operatingMargins",0)*100,1) if d.get("operatingMargins") is not None else None,
            })
        return pd.DataFrame(rows).set_index("Symbol")

    ov_df = build_fin_display(st.session_state.overview_tickers)

    def _cell(val, good, bad, invert=False):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "color:#555555"
        if invert:
            return "color:#4CAF50" if val < good else ("color:#C8102E" if val > bad else "color:#FFC107")
        return "color:#4CAF50" if val > good else ("color:#C8102E" if val < bad else "color:#FFC107")

    styled_ov = (
        ov_df.style
        .map(lambda v: _cell(v,15,0),        subset=["ROE (%)"])
        .map(lambda v: _cell(v,8,0),         subset=["ROA (%)"])
        .map(lambda v: _cell(v,0,100,True),  subset=["D/E"])
        .map(lambda v: _cell(v,2.0,1.0),     subset=["Current Ratio"])
        .map(lambda v: _cell(v,10,0),        subset=["Profit Margin (%)"])
        .map(lambda v: _cell(v,15,0),        subset=["EPS Growth (%)"])
        .map(lambda v: _cell(v,8,0),         subset=["Rev Growth (%)"])
        .map(lambda v: _cell(v,30,0),        subset=["Gross Margin (%)"])
        .map(lambda v: _cell(v,10,0),        subset=["Op Margin (%)"])
        .format({
            "Price ($)":         lambda x: f"${x:.2f}" if pd.notna(x) else "--",
            "ROE (%)":           lambda x: f"{x:.1f}%" if pd.notna(x) else "--",
            "ROA (%)":           lambda x: f"{x:.1f}%" if pd.notna(x) else "--",
            "D/E":               lambda x: f"{x:.1f}"  if pd.notna(x) else "--",
            "Current Ratio":     lambda x: f"{x:.2f}"  if pd.notna(x) else "--",
            "Profit Margin (%)": lambda x: f"{x:.1f}%" if pd.notna(x) else "--",
            "EPS Growth (%)":    lambda x: f"{x:.1f}%" if pd.notna(x) else "--",
            "Rev Growth (%)":    lambda x: f"{x:.1f}%" if pd.notna(x) else "--",
            "Gross Margin (%)":  lambda x: f"{x:.1f}%" if pd.notna(x) else "--",
            "Op Margin (%)":     lambda x: f"{x:.1f}%" if pd.notna(x) else "--",
        })
        .set_properties(**{"background-color":"#2A2A2A","color":"#E8E8E8",
                           "border":"1px solid #3D3D3D","font-size":"0.84rem"})
        .set_table_styles([
            {"selector":"th","props":[("background-color","#000000"),("color","#C8102E"),
                                      ("font-size","0.76rem"),("text-align","center"),
                                      ("padding","8px"),("font-weight","700"),
                                      ("letter-spacing","0.05em"),("text-transform","uppercase")]},
            {"selector":"td","props":[("text-align","center"),("padding","6px 10px")]},
        ])
    )
    st.markdown(
        f"<p style='color:#A0A0A0;font-size:0.8rem;'>"
        f"Portfolio: {len(st.session_state.overview_tickers)} stocks &nbsp;|&nbsp; "
        f"<span style='color:#4CAF50;'>Green</span> = above threshold &nbsp; "
        f"<span style='color:#C8102E;'>Red</span> = below threshold &nbsp; "
        f"<span style='color:#FFC107;'>Yellow</span> = moderate</p>",
        unsafe_allow_html=True)
    st.dataframe(styled_ov, width='stretch', height=530)

    st.divider()
    with st.expander("Lynch's PEG Ratio Guide"):
        st.markdown("""
**PEG Ratio** = P/E / Earnings Growth Rate

| PEG Value | Lynch's Interpretation |
|-----------|----------------------|
| < 0.5     | Hidden gem -- potentially a ten-bagger |
| 0.5 - 1.0 | Bargain -- good risk/reward |
| 1.0 - 2.0 | Fairly priced |
| > 2.0     | Expensive |
        """)


# =============================================================================
# TAB 4 -- K-MEANS STOCK SCREENER
# Strictly follows professor's notebook:
#   yahooquery Ticker.financial_data -> fin_data_df.T -> fillna(0) ->
#   drop non-numeric cols -> MinMaxScaler -> KMeans(4, random_state=100)
#   Long = cluster 3,  Short = cluster 2
# =============================================================================
with tab_kmeans:
    st.markdown(
        "<h2 style='margin-bottom:0.2rem;'>K-Means Stock Screener</h2>"
        "<p style='color:#A0A0A0;font-size:0.9rem;'>"
        "Replicates professor's K-Means algorithm exactly using yahooquery. "
        "Cluster 3 = Long, Cluster 2 = Short. Default: Dow Jones 30.</p>",
        unsafe_allow_html=True)
    st.divider()

    DROP_COLS = [
        "maxAge","currentPrice","targetHighPrice","targetLowPrice",
        "targetMeanPrice","targetMedianPrice","recommendationMean",
        "recommendationKey","numberOfAnalystOpinions","financialCurrency",
    ]

    col_l, col_r = st.columns([4, 1])
    with col_l:
        ticker_input_km = st.text_input(
            "Stock universe (comma-separated -- Dow Jones 30 by default)",
            value=", ".join(DOW30), key="km_tickers")
    with col_r:
        st.markdown("<br>", unsafe_allow_html=True)
        run_km = st.button("Run K-Means", width='stretch', key="run_km_btn")

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_fin_data_yahooquery(tickers):
        live_cols = {}
        try:
            from yahooquery import Ticker
            obj = Ticker(list(tickers))
            raw = obj.financial_data
            if isinstance(raw, dict):
                for sym, data in raw.items():
                    if isinstance(data, dict) and len(data) > 3:
                        live_cols[sym] = data
        except Exception:
            pass
        combined = {}
        live_count = 0
        for t in tickers:
            if t in live_cols:
                combined[t] = live_cols[t]
                live_count += 1
            elif t in FIN_DATA_STATIC:
                combined[t] = FIN_DATA_STATIC[t]
            else:
                combined[t] = {}
        static_used = len(tickers) - live_count
        if static_used > 0:
            st.info(f"Using cached data for {static_used} stock(s). Live data for {live_count} stock(s).")
        return pd.DataFrame.from_dict(combined, orient="index").T

    def run_professor_kmeans(fin_data_df):
        data_full = fin_data_df.T
        data = data_full.fillna(0)
        drop = [c for c in DROP_COLS if c in data.columns]
        numeric_data = data.drop(columns=drop, errors="ignore")
        numeric_data = numeric_data.apply(pd.to_numeric, errors="coerce").fillna(0)
        if numeric_data.shape[0] < 4 or numeric_data.shape[1] == 0:
            return None, None, [], [], numeric_data
        X = numeric_data.to_numpy()
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        model = KMeans(n_clusters=4, random_state=100, n_init=10)
        model.fit(X_scaled)
        yhat = model.predict(X_scaled)
        long_stocks  = list(numeric_data.index[yhat == 3])
        short_stocks = list(numeric_data.index[yhat == 2])
        return X_scaled, yhat, long_stocks, short_stocks, numeric_data

    if run_km:
        tickers_km = tuple(t.strip().upper() for t in ticker_input_km.split(",") if t.strip())
        if len(tickers_km) < 4:
            st.warning("Please enter at least 4 tickers.")
        else:
            with st.spinner(f"Fetching financial data for {len(tickers_km)} stocks via yahooquery..."):
                fin_df = fetch_fin_data_yahooquery(tickers_km)
            with st.spinner("Running K-Means..."):
                X_scaled, yhat, long_list, short_list, numeric_data = run_professor_kmeans(fin_df)

            if X_scaled is None:
                st.error("Not enough valid data to run clustering.")
            else:
                st.success(f"K-Means complete -- {len(tickers_km)} stocks in 4 clusters")

                with st.expander("Methodology (follows professor's notebook exactly)"):
                    st.markdown("""
**Library:** `yahooquery` -- same as professor's `from yahooquery import Ticker`

**Steps (professor's exact code):**
1. `ticker = Ticker(stock_symbol_list)`
2. `fin_data_df = pd.DataFrame.from_dict(ticker.financial_data, orient='index').T`
3. `data = fin_data_df.T.fillna(0)`
4. Drop non-numeric columns, convert to numpy
5. `scaler = MinMaxScaler()` -- scale to [0,1]
6. `model = KMeans(n_clusters=4, random_state=100)`
7. `Long Stocks  = data.index[yhat == 3]`
8. `Short Stocks = data.index[yhat == 2]`
                    """)

                st.divider()
                st.markdown("#### Value-Quality Cluster Chart")

                C = {0:"#A0A0A0", 1:"#FFC107", 2:"#C8102E", 3:"#4CAF50"}
                L = {0:"Neutral A", 1:"Neutral B", 2:"Short", 3:"Long"}
                fig = go.Figure()
                tl = list(numeric_data.index)
                for cid in range(4):
                    ix = [i for i, m in enumerate(yhat == cid) if m]
                    if not ix:
                        continue
                    fig.add_trace(go.Scatter(
                        x=X_scaled[ix, 0], y=X_scaled[ix, 1],
                        mode="markers+text", name=L[cid],
                        text=[tl[i] for i in ix],
                        textposition="top center",
                        textfont=dict(size=10, color="#E8E8E8"),
                        marker=dict(size=14, color=C[cid], line=dict(color="#1A1A1A", width=1.5)),
                        hovertemplate=f"<b>%{{text}}</b><br>Cluster: {L[cid]}<br>"
                                      "Value: %{x:.3f}<br>Quality: %{y:.3f}<extra></extra>"))
                fig.update_layout(
                    paper_bgcolor="#1A1A1A", plot_bgcolor="#2A2A2A",
                    font=dict(color="#E8E8E8", family="Lato"),
                    legend=dict(bgcolor="#2A2A2A", bordercolor="#3D3D3D", font=dict(size=12)),
                    xaxis=dict(title="Value", gridcolor="#3D3D3D", zeroline=False),
                    yaxis=dict(title="Quality", gridcolor="#3D3D3D", zeroline=False),
                    margin=dict(l=10, r=10, t=20, b=10), height=480)
                st.plotly_chart(fig, width='stretch')

                st.divider()
                long_col, short_col = st.columns(2)
                with long_col:
                    st.markdown("<h3 style='color:#4CAF50;'>Long Stocks</h3>", unsafe_allow_html=True)
                    if long_list:
                        for t in long_list:
                            d = FIN_DATA_STATIC.get(t, {})
                            eg  = d.get("earningsGrowth")
                            roe = d.get("returnOnEquity")
                            st.markdown(
                                f"<div class='long-card'>"
                                f"<b style='color:#4CAF50;font-size:1.05rem;'>{t}</b><br>"
                                f"<span style='font-size:0.82rem;color:#A0A0A0;'>"
                                f"EPS Growth: <b>{round(eg*100,1) if eg else 'N/A'}%</b>"
                                f" | ROE: <b>{round(roe*100,1) if roe else 'N/A'}%</b>"
                                f"</span></div>", unsafe_allow_html=True)
                    else:
                        st.info("No Long stocks in this run.")

                with short_col:
                    st.markdown("<h3 style='color:#C8102E;'>Short Stocks</h3>", unsafe_allow_html=True)
                    if short_list:
                        for t in short_list:
                            d = FIN_DATA_STATIC.get(t, {})
                            eg  = d.get("earningsGrowth")
                            roe = d.get("returnOnEquity")
                            st.markdown(
                                f"<div class='short-card'>"
                                f"<b style='color:#C8102E;font-size:1.05rem;'>{t}</b><br>"
                                f"<span style='font-size:0.82rem;color:#A0A0A0;'>"
                                f"EPS Growth: <b>{round(eg*100,1) if eg else 'N/A'}%</b>"
                                f" | ROE: <b>{round(roe*100,1) if roe else 'N/A'}%</b>"
                                f"</span></div>", unsafe_allow_html=True)
                    else:
                        st.info("No Short stocks in this run.")

                st.divider()
                st.markdown("#### Full Financial Data (fin_data_df)")
                disp_cols = ["earningsGrowth","revenueGrowth","returnOnEquity",
                             "returnOnAssets","grossMargins","operatingMargins",
                             "profitMargins","debtToEquity","currentRatio"]
                show_df = numeric_data[[c for c in disp_cols if c in numeric_data.columns]].copy()

                def hl(row):
                    if row.name in long_list:
                        return ["background-color:#1A2A1A;color:#4CAF50"] * len(row)
                    if row.name in short_list:
                        return ["background-color:#2A1A1A;color:#C8102E"] * len(row)
                    return ["background-color:#2A2A2A;color:#E8E8E8"] * len(row)

                st.dataframe(
                    show_df.style.apply(hl, axis=1).format("{:.4f}", na_rep="--")
                    .set_table_styles([
                        {"selector":"th","props":[("background-color","#000000"),
                                                  ("color","#C8102E"),("font-size","0.74rem"),
                                                  ("text-align","center"),("padding","6px")]},
                        {"selector":"td","props":[("font-size","0.82rem"),
                                                  ("text-align","center"),("padding","5px 8px")]},
                    ]),
                    width='stretch', height=420)
    else:
        st.markdown(
            "<div style='text-align:center;padding:4rem 0;color:#555555;font-style:italic;'>"
            "Configure the stock universe above and click <b>Run K-Means</b> to generate signals."
            "</div>", unsafe_allow_html=True)
