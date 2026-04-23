"""
app.py  ──  Peter Lynch ChatBot  ──  Streamlit UI
═══════════════════════════════════════════════════
Part I  : Chat with Peter Lynch (RAG-powered)
Part II : Lynch Stock Analyzer  (PEG ratio + fundamentals + backtest)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from lynch_rag import ask_lynch, load_pipeline
import stock_cache

# ─────────────────────────────────────────────────────────────────────────────
# Page config & global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Peter Lynch · ChatBot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Source+Sans+3:wght@300;400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: #0d1117;
    color: #e6e6e6;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1923 0%, #0d1117 100%);
    border-right: 1px solid #1e2d3d;
}

/* ── Headlines ── */
h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #f0c040;
    letter-spacing: -0.02em;
}

/* ── Chat bubbles ── */
.chat-user {
    background: #1a2433;
    border-left: 3px solid #f0c040;
    border-radius: 0 12px 12px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.95rem;
}
.chat-lynch {
    background: #12233a;
    border-left: 3px solid #2d9cdb;
    border-radius: 0 12px 12px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.95rem;
    line-height: 1.65;
}
.speaker-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.user-label  { color: #f0c040; }
.lynch-label { color: #2d9cdb; }

/* ── Metric cards ── */
.metric-card {
    background: #12233a;
    border: 1px solid #1e2d3d;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
.metric-card .value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f0c040;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #8899aa;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── PEG verdict badge ── */
.verdict-undervalued { color: #27ae60; font-weight: 700; font-size:1.1rem; }
.verdict-fair        { color: #f0c040; font-weight: 700; font-size:1.1rem; }
.verdict-overvalued  { color: #e74c3c; font-weight: 700; font-size:1.1rem; }

/* ── Input ── */
.stTextInput > div > div > input {
    background: #12233a !important;
    color: #e6e6e6 !important;
    border: 1px solid #1e2d3d !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: #f0c040 !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover {
    background: #ffd966 !important;
}

/* ── Tabs ── */
.stTabs [role="tab"] {
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    color: #8899aa;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    color: #f0c040 !important;
    border-bottom: 2px solid #f0c040 !important;
    background: transparent !important;
}

/* ── Divider ── */
hr { border-color: #1e2d3d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h1 style='font-size:1.8rem; margin-bottom:0;'>Peter Lynch</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#8899aa; font-size:0.85rem; margin-top:4px;'>"
        "Fidelity Magellan Fund · 1977–1990<br>"
        "29.2% avg. annual return</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Lynch's Core Principles**")
    principles = [
        "📌 Invest in what you know",
        "📌 PEG < 1 = Bargain",
        "📌 Avoid the 'whisper stock'",
        "📌 Know why you own a stock",
        "📌 Patience beats market-timing",
        "📌 Boring companies can be great",
    ]
    for p in principles:
        st.markdown(f"<div style='font-size:0.82rem; padding:3px 0; color:#c0ccd8;'>{p}</div>",
                    unsafe_allow_html=True)

    st.divider()

    # Pipeline loader
    if not st.session_state.pipeline_ready:
        if st.button("🚀 Load Lynch's Knowledge Base"):
            with st.spinner("Indexing documents…"):
                try:
                    load_pipeline()
                    st.session_state.pipeline_ready = True
                    st.success("Ready!")
                except Exception as exc:
                    st.error(f"Error: {exc}")
    else:
        st.success("✅ Knowledge base loaded")
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown(
        "<p style='font-size:0.75rem; color:#556677;'>"
        "Powered by LangChain · ChromaDB · Groq (Llama 3)</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main content  ── two tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_stocks = st.tabs(["Chat with Peter Lynch", "Lynch Stock Analyzer"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 – CHAT
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown(
        "<h2 style='margin-bottom:0.2rem;'>Ask Peter Lynch</h2>"
        "<p style='color:#8899aa; font-size:0.9rem;'>"
        "Ask about investment strategy, stock selection, portfolio management, or his career.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Conversation history ──────────────────────────────────────────────────
    history_container = st.container()
    with history_container:
        if not st.session_state.chat_history:
            st.markdown(
                "<div style='color:#4a5e70; font-style:italic; text-align:center; "
                "padding:3rem 0;'>Start the conversation below ↓</div>",
                unsafe_allow_html=True,
            )
        else:
            for turn in st.session_state.chat_history:
                st.markdown(
                    f"<div class='speaker-label user-label'>You</div>"
                    f"<div class='chat-user'>{turn['user']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='speaker-label lynch-label'>Peter Lynch</div>"
                    f"<div class='chat-lynch'>{turn['assistant']}</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Input row ─────────────────────────────────────────────────────────────
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_question = st.text_input(
            label="Your question",
            placeholder="e.g. How do you use the PEG ratio to find bargains?",
            label_visibility="collapsed",
            key="chat_input",
        )
    with col_btn:
        send = st.button("Ask →", use_container_width=True)

    # ── Suggested questions ───────────────────────────────────────────────────
    st.markdown(
        "<p style='color:#556677; font-size:0.8rem; margin-top:8px;'>Try asking:</p>",
        unsafe_allow_html=True,
    )
    suggestions = [
        "What is the PEG ratio?",
        "How do you identify a ten-bagger?",
        "What sectors do you prefer?",
        "How do you size a position?",
    ]
    cols = st.columns(len(suggestions))
    for col, sug in zip(cols, suggestions):
        with col:
            if st.button(sug, key=f"sug_{sug}", use_container_width=True):
                user_question = sug
                send = True

    # ── Handle submission ─────────────────────────────────────────────────────
    if send and user_question.strip():
        if not st.session_state.pipeline_ready:
            st.warning("Please load the knowledge base first (sidebar → 🚀).")
        else:
            with st.spinner("Peter Lynch is thinking…"):
                answer = ask_lynch(
                    question=user_question,
                    history=st.session_state.chat_history,
                )
            st.session_state.chat_history.append(
                {"user": user_question, "assistant": answer}
            )
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 – STOCK ANALYZER
# ═════════════════════════════════════════════════════════════════════════════
with tab_stocks:
    st.markdown(
        "<h2 style='margin-bottom:0.2rem;'>Lynch Stock Analyzer</h2>"
        "<p style='color:#8899aa; font-size:0.9rem;'>"
        "Evaluate stocks through Peter Lynch's PEG-ratio lens and backtest his strategy.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Ticker input ──────────────────────────────────────────────────────────
    col_t, col_b = st.columns([4, 1])
    with col_t:
        ticker_input = st.text_input(
            "Enter ticker(s) separated by commas",
            placeholder="e.g.  AAPL, MSFT, WMT, KO",
            label_visibility="collapsed",
        )
    with col_b:
        analyze_btn = st.button("Analyze 🔍", use_container_width=True)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def peg_verdict(peg: float | None) -> str:
        if peg is None or np.isnan(peg):
            return "<span style='color:#8899aa;'>N/A</span>"
        if peg < 1.0:
            return f"<span class='verdict-undervalued'>✅ Bargain ({peg:.2f})</span>"
        if peg <= 2.0:
            return f"<span class='verdict-fair'>⚖️ Fair ({peg:.2f})</span>"
        return f"<span class='verdict-overvalued'>❌ Expensive ({peg:.2f})</span>"


    def safe_get(info: dict, *keys, default=None):
        for k in keys:
            v = info.get(k)
            if v is not None:
                return v
        return default


    def fetch_fundamentals(ticker: str) -> dict:
        # ── Attempt live Yahoo Finance data ──────────────────────────────────
        try:
            stk  = yf.Ticker(ticker)
            info = stk.info

            pe         = safe_get(info, "trailingPE", "forwardPE")
            eps_growth = safe_get(info, "earningsGrowth", "revenueGrowth")
            if eps_growth is not None:
                eps_growth *= 100

            peg = None
            if pe and eps_growth and eps_growth > 0:
                peg = pe / eps_growth

            market_cap = info.get("marketCap")
            if market_cap:
                if market_cap >= 1e12:
                    mc_str = f"${market_cap/1e12:.1f}T"
                elif market_cap >= 1e9:
                    mc_str = f"${market_cap/1e9:.1f}B"
                else:
                    mc_str = f"${market_cap/1e6:.0f}M"
            else:
                mc_str = "N/A"

            price      = info.get("currentPrice") or info.get("regularMarketPrice")
            shares     = info.get("sharesOutstanding")
            cash_total = info.get("totalCash")
            cash_per_share = (cash_total / shares) if cash_total and shares else None

            fcf         = info.get("freeCashflow")
            ps_ratio    = info.get("priceToSalesTrailing12Months")
            current_r   = info.get("currentRatio")
            inst_own    = info.get("institutionalOwnershipPercent") or info.get("heldPercentInstitutions")
            insider_own = info.get("heldPercentInsiders")
            revenue_g   = info.get("revenueGrowth")
            if revenue_g is not None:
                revenue_g *= 100

            def classify():
                if eps_growth is None:
                    return "Unknown", "#8899aa"
                if eps_growth < 5:
                    return "Slow Grower", "#8899aa"
                if eps_growth < 15:
                    return "Stalwart", "#2d9cdb"
                if eps_growth >= 20:
                    return "Fast Grower 🚀", "#27ae60"
                return "Moderate Grower", "#f0c040"

            category, cat_color = classify()

            # Treat a nearly-empty info dict as a failure so we fall back to cache
            if not price and pe is None and eps_growth is None:
                raise ValueError("live API returned no usable data")

            return {
                "name":          info.get("longName", ticker),
                "sector":        info.get("sector", "N/A"),
                "industry":      info.get("industry", "N/A"),
                "price":         price,
                "pe":            pe,
                "eps_growth":    eps_growth,
                "revenue_growth": revenue_g,
                "peg":           peg,
                "market_cap":    mc_str,
                "debt_equity":   info.get("debtToEquity"),
                "roe":           info.get("returnOnEquity"),
                "dividend":      info.get("dividendYield"),
                "cash_per_share": cash_per_share,
                "fcf":           fcf,
                "ps_ratio":      ps_ratio,
                "current_ratio": current_r,
                "inst_own":      inst_own,
                "insider_own":   insider_own,
                "category":      category,
                "cat_color":     cat_color,
                "description":   info.get("longBusinessSummary", ""),
            }
        except Exception:
            pass

        # ── Cache fallback: derive what we can from local OHLCV data ─────────
        cached = stock_cache.compute_fundamentals(ticker)
        if cached is not None:
            return cached

        return {"error": f"No data available for {ticker}. The live API is unreachable and this ticker is not in the local cache.", "name": ticker}


    def fetch_history(ticker: str, period: str = "5y") -> pd.DataFrame:
        # ── Attempt live Yahoo Finance data ──────────────────────────────────
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                return df
        except Exception:
            pass

        # ── Cache fallback ────────────────────────────────────────────────────
        return stock_cache.load_history(ticker)


    def lynch_backtest(hist: pd.DataFrame, pe_buy: float = 15.0, pe_sell: float = 25.0) -> pd.DataFrame:
        """
        Simple Lynch-inspired strategy:
          - Buy when 52-week P/E equivalent (price below 200-day SMA × buy_mult) 
          - Actually: buy when price crosses above 50-day MA (momentum)
            AND P/E estimate below threshold  →  simplified to MA cross + RSI proxy.
        We approximate 'Lynch value zone' with:
          Buy signal  : Close > 200d MA  AND  Close < 1.05 × 200d MA   (entering value zone)
          Sell signal : Close > 1.30 × 200d MA                          (extended)
        Returns a DataFrame with signals and cumulative returns.
        """
        if hist.empty or len(hist) < 250:
            return pd.DataFrame()

        # Flatten multi-level columns from yfinance (e.g. ('Close','AAPL') → 'Close')
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        df = pd.DataFrame(index=hist.index)
        df["Close"] = hist["Close"].squeeze().astype(float)
        df["MA200"] = df["Close"].rolling(200).mean()
        df["MA50"]  = df["Close"].rolling(50).mean()

        df["signal"] = 0
        in_trade = False
        for i in range(200, len(df)):
            close  = float(df["Close"].iloc[i])
            ma200  = float(df["MA200"].iloc[i])
            if not in_trade and close > ma200 and close < ma200 * 1.10:
                df.iloc[i, df.columns.get_loc("signal")] = 1   # BUY
                in_trade = True
            elif in_trade and close > ma200 * 1.30:
                df.iloc[i, df.columns.get_loc("signal")] = -1  # SELL
                in_trade = False

        # Cumulative returns
        df["daily_ret"]  = df["Close"].pct_change()
        df["position"]   = df["signal"].replace(0, np.nan).ffill().fillna(0).clip(lower=0)
        df["strat_ret"]  = df["position"] * df["daily_ret"]
        df["strat_cum"]  = (1 + df["strat_ret"]).cumprod()
        df["bnh_cum"]    = (1 + df["daily_ret"]).cumprod()
        return df.dropna(subset=["MA200"])

    # ── Analysis output ───────────────────────────────────────────────────────
    if analyze_btn and ticker_input.strip():
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

        for ticker in tickers:
            with st.spinner(f"Fetching data for {ticker}…"):
                fund = fetch_fundamentals(ticker)
                hist = fetch_history(ticker)

            if "error" in fund:
                st.error(f"**{ticker}**: {fund['error']}")
                continue

            if fund.get("from_cache"):
                st.info(
                    f"Live data unavailable for **{ticker}** (Yahoo Finance rate-limit). "
                    "Showing historical data from local cache(2020.1-2026.4)"
                )

            # ── Company header ────────────────────────────────────────────────
            st.markdown(
                f"<h3 style='margin-top:1.5rem;'>{fund['name']} "
                f"<span style='color:#556677; font-size:1rem;'>({ticker})</span></h3>"
                f"<p style='color:#8899aa; font-size:0.85rem; margin-top:-8px;'>"
                f"{fund['sector']} · {fund['industry']}</p>",
                unsafe_allow_html=True,
            )

            # ── Company category badge ────────────────────────────────────────
            st.markdown(
                f"<span style='background:{fund['cat_color']}22; color:{fund['cat_color']}; "
                f"border:1px solid {fund['cat_color']}55; border-radius:6px; "
                f"padding:3px 10px; font-size:0.8rem; font-weight:600;'>"
                f"Lynch Category: {fund['category']}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # ── Metric row (row 1) ────────────────────────────────────────────
            m1, m2, m3, m4, m5 = st.columns(5)
            price_str = f"${fund['price']:.2f}" if fund["price"] else "N/A"
            pe_str    = f"{fund['pe']:.1f}×"    if fund["pe"]    else "N/A"
            epsg_str  = f"{fund['eps_growth']:.1f}%" if fund["eps_growth"] is not None else "N/A"
            roe_str   = f"{fund['roe']*100:.1f}%"    if fund["roe"] is not None else "N/A"
            div_str   = f"{fund['dividend']*100:.2f}%" if fund["dividend"] else "0%"

            for col, val, lbl in zip(
                [m1, m2, m3, m4, m5],
                [price_str, pe_str, epsg_str, roe_str, div_str],
                ["Price", "P/E Ratio", "EPS Growth", "ROE", "Dividend Yield"],
            ):
                col.markdown(
                    f"<div class='metric-card'><div class='value'>{val}</div>"
                    f"<div class='label'>{lbl}</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Metric row (row 2) ────────────────────────────────────────────
            n1, n2, n3, n4, n5 = st.columns(5)
            peg_str  = f"{fund['peg']:.2f}"        if fund["peg"]           else "N/A"
            de_str   = f"{fund['debt_equity']:.0f}%" if fund["debt_equity"]  else "N/A"
            ps_str   = f"{fund['ps_ratio']:.1f}×"  if fund["ps_ratio"]      else "N/A"
            cr_str   = f"{fund['current_ratio']:.1f}x" if fund["current_ratio"] else "N/A"
            ins_str  = f"{fund['insider_own']*100:.1f}%" if fund["insider_own"] else "N/A"

            for col, val, lbl in zip(
                [n1, n2, n3, n4, n5],
                [peg_str, de_str, ps_str, cr_str, ins_str],
                ["PEG Ratio", "Debt/Equity", "P/S Ratio", "Current Ratio", "Insider Ownership"],
            ):
                col.markdown(
                    f"<div class='metric-card'><div class='value'>{val}</div>"
                    f"<div class='label'>{lbl}</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Lynch Full Scorecard ──────────────────────────────────────────
            score_col, desc_col = st.columns([3, 2])

            with score_col:
                st.markdown("**Lynch Full Scorecard**")

                def check(label, value, good, warn, bad, fmt=None, invert=False):
                    """Render a single scorecard row with pass/warn/fail."""
                    if value is None:
                        icon, color, display = "⚪", "#8899aa", "N/A"
                    else:
                        display = fmt(value) if fmt else str(value)
                        if invert:
                            if value <= good:   icon, color = "✅", "#27ae60"
                            elif value <= warn: icon, color = "⚠️", "#f0c040"
                            else:               icon, color = "❌", "#e74c3c"
                        else:
                            if value >= good:   icon, color = "✅", "#27ae60"
                            elif value >= warn: icon, color = "⚠️", "#f0c040"
                            else:               icon, color = "❌", "#e74c3c"
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between; "
                        f"padding:5px 8px; margin:3px 0; background:#12233a; "
                        f"border-radius:6px; font-size:0.85rem;'>"
                        f"<span style='color:#c0ccd8;'>{icon} {label}</span>"
                        f"<span style='color:{color}; font-weight:600;'>{display}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # PEG
                check("PEG Ratio (< 1 = bargain)", fund["peg"],
                      good=1.0, warn=2.0, bad=99,
                      fmt=lambda v: f"{v:.2f}", invert=True)

                # EPS Growth
                check("EPS Growth (> 15% = fast grower)", fund["eps_growth"],
                      good=15, warn=5, bad=0,
                      fmt=lambda v: f"{v:.1f}%")

                # Debt/Equity (lower is better)
                check("Debt/Equity (< 80% = healthy)", fund["debt_equity"],
                      good=40, warn=80, bad=999,
                      fmt=lambda v: f"{v:.0f}%", invert=True)

                # ROE
                check("ROE (> 15% = strong)", fund["roe"] * 100 if fund["roe"] else None,
                      good=15, warn=10, bad=0,
                      fmt=lambda v: f"{v:.1f}%")

                # Current ratio
                check("Current Ratio (> 2 = safe)", fund["current_ratio"],
                      good=2.0, warn=1.0, bad=0,
                      fmt=lambda v: f"{v:.1f}x")

                # Insider ownership
                check("Insider Ownership (> 5% = skin in game)",
                      fund["insider_own"] * 100 if fund["insider_own"] else None,
                      good=5, warn=1, bad=0,
                      fmt=lambda v: f"{v:.1f}%")

                # Institutional ownership (Lynch liked LOW — undiscovered)
                inst_pct = fund["inst_own"] * 100 if fund["inst_own"] else None
                check("Institutional Ownership (< 50% = undiscovered)",
                      inst_pct,
                      good=50, warn=75, bad=999,
                      fmt=lambda v: f"{v:.1f}%", invert=True)

                # P/S ratio
                check("P/S Ratio (< 2 = reasonable)", fund["ps_ratio"],
                      good=1.0, warn=2.0, bad=99,
                      fmt=lambda v: f"{v:.1f}×", invert=True)

                # Lynch verdict
                scores = []
                if fund["peg"] and fund["peg"] < 1:           scores.append(1)
                if fund["eps_growth"] and fund["eps_growth"] > 15: scores.append(1)
                if fund["debt_equity"] and fund["debt_equity"] < 80: scores.append(1)
                if fund["roe"] and fund["roe"] > 0.15:         scores.append(1)
                if fund["current_ratio"] and fund["current_ratio"] > 2: scores.append(1)
                if fund["insider_own"] and fund["insider_own"] > 0.05:  scores.append(1)

                n_pass = len(scores)
                if n_pass >= 5:
                    verdict = "🏆 Strong Lynch Pick"
                    vc = "#27ae60"
                elif n_pass >= 3:
                    verdict = "👍 Decent Candidate"
                    vc = "#f0c040"
                else:
                    verdict = "👎 Doesn't Pass Lynch's Test"
                    vc = "#e74c3c"

                st.markdown(
                    f"<div style='margin-top:12px; padding:10px 14px; "
                    f"background:{vc}22; border:1px solid {vc}55; border-radius:8px; "
                    f"font-weight:700; color:{vc}; font-size:0.95rem;'>"
                    f"{verdict} &nbsp;·&nbsp; {n_pass}/6 criteria met</div>",
                    unsafe_allow_html=True,
                )

            with desc_col:
                if fund["description"]:
                    st.markdown("**About the Company**")
                    st.markdown(
                        f"<div style='font-size:0.83rem; color:#b0c0d0; "
                        f"line-height:1.6;'>{fund['description'][:500]}…</div>",
                        unsafe_allow_html=True,
                    )

            # ── Price chart + backtest ────────────────────────────────────────
            if not hist.empty:
                bt = lynch_backtest(hist)

                chart_col, bt_col = st.columns([3, 2])

                with chart_col:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist.index.tolist(),
                        y=hist["Close"].squeeze().astype(float).tolist(),
                        mode="lines", name="Price",
                        line=dict(color="#2d9cdb", width=1.5),
                    ))
                    if not bt.empty:
                        fig.add_trace(go.Scatter(
                            x=bt.index.tolist(), y=bt["MA200"].tolist(),
                            mode="lines", name="200-day MA",
                            line=dict(color="#f0c040", width=1, dash="dot"),
                        ))
                        fig.add_trace(go.Scatter(
                            x=bt.index.tolist(), y=bt["MA50"].tolist(),
                            mode="lines", name="50-day MA",
                            line=dict(color="#e67e22", width=1, dash="dot"),
                        ))
                        buys  = bt[bt["signal"] ==  1]
                        sells = bt[bt["signal"] == -1]
                        if not buys.empty:
                            fig.add_trace(go.Scatter(
                                x=buys.index.tolist(), y=buys["Close"].tolist(),
                                mode="markers", name="Buy Signal",
                                marker=dict(color="#27ae60", symbol="triangle-up", size=10),
                            ))
                        if not sells.empty:
                            fig.add_trace(go.Scatter(
                                x=sells.index.tolist(), y=sells["Close"].tolist(),
                                mode="markers", name="Sell Signal",
                                marker=dict(color="#e74c3c", symbol="triangle-down", size=10),
                            ))
                    fig.update_layout(
                        title=f"{ticker} — 5-Year Price + Moving Averages",
                        paper_bgcolor="#0d1117",
                        plot_bgcolor="#0d1117",
                        font=dict(color="#e6e6e6", family="Source Sans 3"),
                        legend=dict(
                            bgcolor="#12233a",
                            bordercolor="#1e2d3d",
                            font=dict(size=11),
                        ),
                        xaxis=dict(gridcolor="#1e2d3d", showgrid=True),
                        yaxis=dict(gridcolor="#1e2d3d", showgrid=True, title="Price ($)"),
                        margin=dict(l=10, r=10, t=40, b=10),
                        height=320,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with bt_col:
                    if not bt.empty:
                        fig2 = go.Figure()
                        strat_cum = bt["strat_cum"].squeeze()
                        bnh_cum   = bt["bnh_cum"].squeeze()
                        fig2.add_trace(go.Scatter(
                            x=bt.index.tolist(), y=bt["strat_cum"].tolist(),
                            mode="lines", name="Lynch Strategy",
                            line=dict(color="#f0c040", width=2),
                        ))
                        fig2.add_trace(go.Scatter(
                            x=bt.index.tolist(), y=bt["bnh_cum"].tolist(),
                            mode="lines", name="Buy & Hold",
                            line=dict(color="#2d9cdb", width=1.5, dash="dot"),
                        ))
                        fig2.update_layout(
                            title="Strategy vs Buy & Hold",
                            paper_bgcolor="#0d1117",
                            plot_bgcolor="#0d1117",
                            font=dict(color="#e6e6e6", family="Source Sans 3"),
                            legend=dict(
                                bgcolor="#12233a",
                                bordercolor="#1e2d3d",
                                font=dict(size=11),
                            ),
                            xaxis=dict(gridcolor="#1e2d3d"),
                            yaxis=dict(gridcolor="#1e2d3d", title="Cumulative Return"),
                            margin=dict(l=10, r=10, t=40, b=10),
                            height=320,
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                        # Summary stats
                        strat_final = bt["strat_cum"].iloc[-1]
                        bnh_final   = bt["bnh_cum"].iloc[-1]
                        n_trades    = int((bt["signal"] != 0).sum())
                        st.markdown(
                            f"<div style='font-size:0.82rem; color:#8899aa; text-align:center;'>"
                            f"Strategy: <b style='color:#f0c040;'>"
                            f"+{(strat_final-1)*100:.1f}%</b> &nbsp;|&nbsp; "
                            f"Buy & Hold: <b style='color:#2d9cdb;'>"
                            f"+{(bnh_final-1)*100:.1f}%</b> &nbsp;|&nbsp; "
                            f"Signals fired: <b>{n_trades}</b></div>",
                            unsafe_allow_html=True,
                        )

            st.divider()

    elif analyze_btn:
        st.warning("Please enter at least one ticker symbol.")

    # ── PEG legend ────────────────────────────────────────────────────────────
    with st.expander("📖 Lynch's PEG Ratio Guide"):
        st.markdown("""
**The PEG Ratio** = P/E ÷ Earnings Growth Rate

Peter Lynch popularised this metric in *One Up on Wall Street* as a quick way
to spot under- and over-valued growth stocks.

| PEG Value | Lynch's Interpretation |
|-----------|----------------------|
| < 0.5     | Hidden gem — potentially a ten-bagger |
| 0.5 – 1.0 | Bargain — good risk/reward |
| 1.0 – 2.0 | Fairly priced — acceptable if story holds |
| > 2.0     | Expensive — growth must be exceptional |

**Caveats Lynch himself used:**
- Always verify the growth rate is sustainable
- Check the balance sheet (debt kills great companies)
- Understand the business before trusting any ratio
- "In the short run, the market is a voting machine; in the long run, it's a weighing machine."
        """)