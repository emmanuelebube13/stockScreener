import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import datetime
import numpy as np
import warnings
import re
import requests
# FIXED: Added Tuple to imports
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Attempt to import rapidfuzz for fuzzy search, fallback to difflib
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_OK = True
except ImportError:
    import difflib
    RAPIDFUZZ_OK = False

# ==============================================================================
# 0. CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    layout="wide", 
    page_title="Pro Stock Terminal",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS to tighten up spacing and improve metric visibility
st.markdown("""
<style>
    .stMetric {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. CONSTANTS & REGEX
# ==============================================================================

COMMON_INDEX_ALIASES = {
    "SPX": "^GSPC", "S&P500": "^GSPC", "SP500": "^GSPC",
    "NDX": "^NDX", "NASDAQ100": "^NDX",
    "DJI": "^DJI", "DOW": "^DJI",
    "VIX": "^VIX", "DXY": "DX-Y.NYB",
}

COMMON_FUTURES_ALIASES = {
    "GOLD": "GC=F", "SILVER": "SI=F", "OIL": "CL=F", 
    "WTI": "CL=F", "BRENT": "BZ=F", "NATGAS": "NG=F",
    "CORN": "ZC=F", "COPPER": "HG=F"
}

CRYPTO_ALIASES = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
    "DOGE": "DOGE-USD", "XRP": "XRP-USD"
}

# Regex patterns for asset detection
FX_REGEX = re.compile(r"^[A-Z]{3}[\/\-]?[A-Z]{3}$")
CRYPTO_PAIR_REGEX = re.compile(r"^[A-Z0-9]{2,10}[\-\/]?(USD|USDT|EUR)$")
FUTURES_REGEX = re.compile(r"^[A-Z]{1,3}=?F?$")

SEC_HEADERS = {"User-Agent": "StreamlitApp contact@example.com"}

# ==============================================================================
# 2. HELPER FUNCTIONS: FORMATTING
# ==============================================================================

def format_large_number(num: float, currency: str = "$") -> str:
    """
    Formats large numbers into Trillions (T), Billions (B), Millions (M).
    Example: 1,500,000,000 -> $1.50B
    """
    if num is None or not isinstance(num, (int, float)):
        return "N/A"
    
    abs_num = abs(num)
    if abs_num >= 1e12:
        return f"{currency}{num / 1e12:.2f}T"
    elif abs_num >= 1e9:
        return f"{currency}{num / 1e9:.2f}B"
    elif abs_num >= 1e6:
        return f"{currency}{num / 1e6:.2f}M"
    elif abs_num >= 1e3:
        return f"{currency}{num / 1e3:.2f}K"
    else:
        return f"{currency}{num:.2f}"

def format_percentage(num: float) -> str:
    """Formats decimal to percentage (0.05 -> 5.00%)."""
    if num is None or not isinstance(num, (int, float)):
        return "N/A"
    return f"{num * 100:.2f}%"

# ==============================================================================
# 3. HELPER FUNCTIONS: SEARCH & ASSET DETECTION
# ==============================================================================

def detect_and_normalize_instrument(query: str) -> Dict[str, Any]:
    """
    Analyzes the input string to detect asset class and normalize the symbol for Yahoo Finance.
    """
    q = (query or "").strip().upper()
    if not q:
        return {"symbol": "", "asset_type": "unknown", "name": ""}

    # 1. Check known aliases
    if q in COMMON_INDEX_ALIASES:
        return {"symbol": COMMON_INDEX_ALIASES[q], "asset_type": "index", "name": q}
    if q in COMMON_FUTURES_ALIASES:
        return {"symbol": COMMON_FUTURES_ALIASES[q], "asset_type": "future", "name": q}
    if q in CRYPTO_ALIASES:
        return {"symbol": CRYPTO_ALIASES[q], "asset_type": "crypto", "name": q}

    # 2. Check Patterns
    if q.startswith("^"):
        return {"symbol": q, "asset_type": "index", "name": q}
    
    if FX_REGEX.match(q):
        base, quote = q[:3], q[-3:]
        return {"symbol": f"{base}{quote}=X", "asset_type": "fx", "name": f"{base}/{quote}"}

    if "-" in q or "/" in q or CRYPTO_PAIR_REGEX.match(q):
        cleaned = q.replace("/", "-")
        if not any(x in cleaned for x in ["-USD", "-USDT", "-EUR", "-BTC"]):
             if cleaned.endswith("USD"): cleaned = cleaned.replace("USD", "-USD")
        return {"symbol": cleaned, "asset_type": "crypto", "name": cleaned}

    if q.endswith("=F"):
        return {"symbol": q, "asset_type": "future", "name": q}

    # 3. Default to Equity
    return {"symbol": q, "asset_type": "equity", "name": q}

@st.cache_data(ttl=3600)
def yahoo_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Performs a fuzzy search using yfinance and creates a ranked list of candidates.
    """
    q = (query or "").strip()
    if not q: return []

    candidates = []
    
    # Attempt yfinance API search
    try:
        s = yf.Search(q)
        quotes = getattr(s, "quotes", []) or []
        for item in quotes[:max_results]:
            candidates.append({
                "symbol": item.get("symbol", ""),
                "name": item.get("shortname") or item.get("longname") or item.get("quoteType", ""),
                "exch": item.get("exchDisp") or item.get("exchange", ""),
                "type": item.get("quoteType", "EQUITY"),
            })
    except Exception:
        pass

    # Fallback: manual guess
    if not candidates:
        norm = detect_and_normalize_instrument(q)
        if norm["symbol"]:
            candidates.append({
                "symbol": norm["symbol"],
                "name": norm["name"] or norm["symbol"],
                "exch": "Auto",
                "type": norm["asset_type"].upper()
            })

    # Fuzzy Ranking
    for c in candidates:
        text = f"{c['symbol']} {c['name']}".upper()
        if RAPIDFUZZ_OK:
            c["score"] = fuzz.WRatio(q.upper(), text)
        else:
            c["score"] = difflib.SequenceMatcher(None, q.upper(), text).ratio() * 100

    # Sort and Deduplicate
    candidates.sort(key=lambda x: x["score"], reverse=True)
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c["symbol"] not in seen:
            seen.add(c["symbol"])
            unique_candidates.append(c)
            
    return unique_candidates[:max_results]

# ==============================================================================
# 4. DATA FETCHING (ROBUST)
# ==============================================================================

@st.cache_data(ttl=1800)
def fetch_stock_info(symbol: str) -> Dict[str, Any]:
    try:
        t = yf.Ticker(symbol)
        return t.info or {}
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def fetch_financials_robust(symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (Quarterly, Annual) financials transposed properly."""
    try:
        t = yf.Ticker(symbol)
        q = t.quarterly_financials.T if t.quarterly_financials is not None else pd.DataFrame()
        a = t.financials.T if t.financials is not None else pd.DataFrame()
        return q, a
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_price_history(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(symbol)
        df = t.history(start=start, end=end, interval=interval, auto_adjust=False)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_news_robust(symbol: str) -> List[Dict[str, Any]]:
    """Fetches news and filters out empty/broken items."""
    try:
        t = yf.Ticker(symbol)
        raw_news = t.news
        valid_news = []
        if raw_news:
            for n in raw_news:
                # Filter out items with no title or link
                if n.get('title') and n.get('link'):
                    valid_news.append(n)
        return valid_news
    except Exception:
        return []

@st.cache_data(ttl=3600)
def fetch_calendar_robust(symbol: str) -> Any:
    """Safely retrieves calendar data which can be a dict or DataFrame."""
    try:
        t = yf.Ticker(symbol)
        return t.calendar
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_sec_filings(ticker: str) -> pd.DataFrame:
    """Fetch SEC filings via EDGAR (requires mapping ticker to CIK)."""
    try:
        # 1. Get CIK map
        map_url = "https://www.sec.gov/files/company_tickers.json"
        r = requests.get(map_url, headers=SEC_HEADERS, timeout=5)
        if r.status_code != 200: return pd.DataFrame()
        
        cik_map = {v['ticker']: str(v['cik_str']).zfill(10) for k, v in r.json().items()}
        cik = cik_map.get(ticker.upper())
        if not cik: return pd.DataFrame()

        # 2. Get Submissions
        sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r_sub = requests.get(sub_url, headers=SEC_HEADERS, timeout=5)
        if r_sub.status_code != 200: return pd.DataFrame()
        
        data = r_sub.json()
        recent = data.get("filings", {}).get("recent", {})
        if not recent: return pd.DataFrame()
        
        df = pd.DataFrame(recent)
        # Filter for key forms
        df = df[df['form'].isin(['10-K', '10-Q', '8-K'])].head(15).copy()
        
        # Construct URLs
        def build_url(row):
            acc = row['accessionNumber'].replace('-', '')
            doc = row['primaryDocument']
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
        
        df['url'] = df.apply(build_url, axis=1)
        return df[['form', 'reportDate', 'filingDate', 'url']]
    except Exception:
        return pd.DataFrame()

# ==============================================================================
# 5. UI COMPONENTS
# ==============================================================================

def render_sidebar():
    st.sidebar.title("🔍 Asset Navigator")
    
    # --- Search Box ---
    query = st.sidebar.text_input("Symbol / Name", value="TSLA", placeholder="e.g. Tesla, BTC, Gold")
    
    # --- Quick Select Carousel ---
    detected = detect_and_normalize_instrument(query)
    candidates = yahoo_search(query)
    
    # If detection found a non-equity, prioritize it
    if detected["asset_type"] != "equity":
        candidates.insert(0, {
            "symbol": detected["symbol"], 
            "name": detected["name"], 
            "type": detected["asset_type"].upper(),
            "exch": "Auto"
        })

    # Display Top 4 options as buttons
    cols = st.sidebar.columns(4)
    sel_symbol = None
    for i, c in enumerate(candidates[:4]):
        # Shorten label
        lbl = c['symbol'].split("-")[0] if "-" in c['symbol'] and len(c['symbol']) > 8 else c['symbol']
        if cols[i].button(lbl[:6], help=f"{c['name']} ({c['type']})", use_container_width=True):
            sel_symbol = c
            
    # Fallback to text input if no button clicked
    final_symbol = sel_symbol['symbol'] if sel_symbol else detected["symbol"]
    final_type = sel_symbol['type'] if sel_symbol else detected["asset_type"]

    st.sidebar.markdown("---")
    
    # --- Time & Interval ---
    st.sidebar.subheader("⚙️ Chart Settings")
    c1, c2 = st.sidebar.columns(2)
    start_date = c1.date_input("Start", datetime.date.today() - datetime.timedelta(days=365))
    end_date = c2.date_input("End", datetime.date.today())
    
    # Dynamic intervals based on asset type
    if final_type.lower() in ['crypto', 'fx', 'future', 'index']:
        intervals = ["5m", "15m", "30m", "1h", "1d", "1wk"]
        def_idx = 4 # 1d
    else:
        intervals = ["1d", "1wk", "1mo"]
        def_idx = 0
    
    interval = st.sidebar.selectbox("Interval", intervals, index=def_idx)
    
    # --- Compare ---
    compare_ticker = st.sidebar.text_input("Compare Ticker", value="")
    
    return final_symbol, final_type, start_date, end_date, interval, compare_ticker

# ==============================================================================
# 6. CHARTING FUNCTIONS (PLOTLY)
# ==============================================================================

def render_main_chart(df: pd.DataFrame, symbol: str, overlays: List[str] = None):
    """
    Renders a professional candlestick chart with Volume, MA, and RSI/MACD support.
    """
    if df.empty:
        st.warning("No data available for chart.")
        return

    # Calculate Indicators
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Create Subplots (Price + Volume)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.75, 0.25]
    )

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Price'
    ), row=1, col=1)

    # 2. Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1), name='SMA 200'), row=1, col=1)

    # 3. Volume
    colors = ['green' if row['Close'] >= row['Open'] else 'red' for i, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        marker_color=colors,
        name='Volume'
    ), row=2, col=1)

    # Layout Updates
    fig.update_layout(
        title=f"<b>{symbol} Price Action</b>",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=650,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.02, x=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_financial_bar_chart(df: pd.DataFrame, metric: str, title: str, color_seq: str = "blue"):
    """
    Renders a large, colorful bar chart for financial metrics using Plotly.
    """
    if df.empty or metric not in df.columns:
        st.info(f"No data for {title}")
        return

    # Sort by date
    df = df.sort_index()
    
    # Color logic: Green for positive, Red for negative (if Net Income), else Blue
    if "Income" in title or "Profit" in title:
        colors = ['#00C805' if v >= 0 else '#FF3b30' for v in df[metric]]
    else:
        colors = '#1f77b4' # Standard Blue

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index,
        y=df[metric],
        marker_color=colors,
        text=[format_large_number(v) for v in df[metric]],
        textposition='auto',
        name=title
    ))

    fig.update_layout(
        title=f"<b>{title}</b>",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title="Amount ($)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 7. MAIN APPLICATION LOGIC
# ==============================================================================

def main():
    # --- Sidebar Inputs ---
    symbol, asset_type, start_date, end_date, interval, compare_ticker = render_sidebar()

    # --- Data Loading ---
    with st.spinner(f"Loading data for {symbol}..."):
        info = fetch_stock_info(symbol)
        df_price = fetch_price_history(symbol, start_date, end_date, interval)
    
    if df_price.empty:
        st.error(f"No price data found for {symbol}. The ticker might be delisted or invalid.")
        st.stop()

    # --- HEADER SECTION (Formatting Fixes) ---
    name = info.get('longName', info.get('shortName', symbol))
    currency = info.get('currency', 'USD')
    
    # Top Row: Title + Sector
    c1, c2 = st.columns([2, 1])
    with c1:
        st.title(f"{name} ({symbol})")
    with c2:
        st.markdown(f"**Sector:** {info.get('sector','N/A')}  \n**Ind:** {info.get('industry','N/A')}")
    
    st.markdown("---")

    # Metrics Row (Formatted)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    # 1. Price
    curr_price = df_price['Close'].iloc[-1]
    prev_price = df_price['Close'].iloc[-2] if len(df_price) > 1 else curr_price
    delta = curr_price - prev_price
    delta_pct = (delta / prev_price) * 100
    m1.metric("Price", f"{curr_price:,.2f}", f"{delta:+.2f} ({delta_pct:+.2f}%)")
    
    # 2. Market Cap (FIXED: B/T/M formatting)
    mcap = info.get('marketCap')
    m2.metric("Market Cap", format_large_number(mcap), delta_color="off")
    
    # 3. Volume
    vol = df_price['Volume'].iloc[-1]
    m3.metric("Volume", format_large_number(vol, currency=""), delta_color="off")
    
    # 4. P/E
    pe = info.get('trailingPE')
    m4.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
    
    # 5. Dividend
    div = info.get('dividendYield')
    m5.metric("Div Yield", format_percentage(div))
    
    # 6. Beta / Margin
    beta = info.get('beta')
    m6.metric("Beta", f"{beta:.2f}" if beta else "N/A")

    # --- TABS ---
    tabs = st.tabs(["📈 Chart", "📊 Financials", "📰 News", "📅 Events", "🔬 AI Analysis", "🧰 Screener"])

    # ------------------------------------------------------------------
    # TAB 1: CHART
    # ------------------------------------------------------------------
    with tabs[0]:
        render_main_chart(df_price, symbol)
        
        # Overlay Logic
        if compare_ticker:
            st.subheader(f"Comparison: {symbol} vs {compare_ticker}")
            df_comp = fetch_price_history(compare_ticker, start_date, end_date, interval)
            if not df_comp.empty:
                # Normalize to % return
                norm_main = (df_price['Close'] / df_price['Close'].iloc[0] - 1) * 100
                norm_comp = (df_comp['Close'] / df_comp['Close'].iloc[0] - 1) * 100
                
                comp_fig = go.Figure()
                comp_fig.add_trace(go.Scatter(x=norm_main.index, y=norm_main, name=symbol, line=dict(color='blue')))
                comp_fig.add_trace(go.Scatter(x=norm_comp.index, y=norm_comp, name=compare_ticker, line=dict(color='orange')))
                comp_fig.update_layout(title="Relative Performance (%)", yaxis_title="Return %")
                st.plotly_chart(comp_fig, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 2: FINANCIALS (Heavy Rework - Large Charts)
    # ------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Financial Performance")
        
        q_fin, a_fin = fetch_financials_robust(symbol)
        
        period_type = st.radio("Period:", ["Annual", "Quarterly"], horizontal=True)
        active_df = a_fin if period_type == "Annual" else q_fin

        if active_df.empty:
            st.warning(f"No {period_type.lower()} financial data available for {symbol}.")
        else:
            # Ensure index is datetime for proper sorting
            try:
                active_df.index = pd.to_datetime(active_df.index)
            except:
                pass
            
            # Row 1: Revenue & Net Income
            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                render_financial_bar_chart(active_df, "Total Revenue", f"{period_type} Revenue")
            
            with col_f2:
                metric_name = "Net Income" if "Net Income" in active_df.columns else "Net Income Common Stockholders"
                render_financial_bar_chart(active_df, metric_name, f"{period_type} Net Income")
            
            st.markdown("---")
            
            # Row 2: Margins & EBITDA
            if "Total Revenue" in active_df.columns and metric_name in active_df.columns:
                active_df["Net Margin %"] = (active_df[metric_name] / active_df["Total Revenue"]) * 100
                
                fig_margin = go.Figure()
                fig_margin.add_trace(go.Scatter(
                    x=active_df.index, y=active_df["Net Margin %"],
                    mode='lines+markers', name="Net Margin %",
                    line=dict(color='purple', width=3)
                ))
                fig_margin.update_layout(title="Net Profit Margin Trend", yaxis_title="Margin %", height=350)
                st.plotly_chart(fig_margin, use_container_width=True)

            # Raw Data Expander
            with st.expander("View Raw Financial Data"):
                st.dataframe(active_df.style.format("${:,.2f}", na_rep="-"))

    # ------------------------------------------------------------------
    # TAB 3: NEWS (Fixed "None" issue)
    # ------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Latest News")
        news_items = fetch_news_robust(symbol)
        
        if not news_items:
            st.info("No recent news articles found via standard feed.")
        else:
            # Create a nice grid layout
            for i, item in enumerate(news_items[:10]):
                with st.container():
                    c_img, c_text = st.columns([1, 4])
                    
                    # Thumbnail (if available)
                    thumb = item.get('thumbnail', {}).get('resolutions', [])
                    img_url = thumb[0]['url'] if thumb else None
                    
                    with c_img:
                        if img_url:
                            st.image(img_url, use_container_width=True)
                        else:
                            st.write("📰")
                    
                    with c_text:
                        title = item.get('title', 'No Title')
                        link = item.get('link', '#')
                        pub = item.get('publisher', 'Unknown')
                        
                        # Timestamp handling
                        ts = item.get('providerPublishTime')
                        date_str = ""
                        if ts:
                            date_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
                        
                        st.markdown(f"#### [{title}]({link})")
                        st.caption(f"{pub} • {date_str}")
                st.divider()

    # ------------------------------------------------------------------
    # TAB 4: EVENTS (Fixed Formatting)
    # ------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Corporate Calendar")
        
        col_cal, col_fil = st.columns(2)
        
        with col_cal:
            st.markdown("### 🗓 Earnings & Events")
            cal_data = fetch_calendar_robust(symbol)
            
            # Handle different return types from yfinance (dict vs df)
            if isinstance(cal_data, pd.DataFrame) and not cal_data.empty:
                st.dataframe(cal_data, use_container_width=True)
            
            elif isinstance(cal_data, dict) and cal_data:
                # Parse the messy dictionary into a clean table
                rows = []
                for k, v in cal_data.items():
                    # If value is a list (like dates), join them
                    val_str = str(v)
                    if isinstance(v, list):
                        # try to format dates
                        val_str = ", ".join([str(x) for x in v])
                    rows.append({"Event": k, "Value": val_str})
                
                st.table(pd.DataFrame(rows))
            else:
                st.info("No upcoming earnings calendar data found.")

        with col_fil:
            st.markdown("### 🏛 SEC Filings (10-K, 10-Q, 8-K)")
            filings = fetch_sec_filings(symbol)
            
            if filings.empty:
                st.write("No recent filings found (or ticker not US-listed).")
            else:
                # Make clickable links
                for _, row in filings.iterrows():
                    st.markdown(f"**{row['form']}** ({row['filingDate']}) — [View Document]({row['url']})")

    # ------------------------------------------------------------------
    # TAB 5: AI ANALYSIS (Developer/Heavy Logic)
    # ------------------------------------------------------------------
    with tabs[4]:
        st.subheader("🤖 AI Price Prediction Experiment")
        st.info("Note: This uses a RandomForest Classifier on technical indicators (RSI, SMA, Momentum). strictly for educational purposes.")
        
        # Prepare Data
        ml_df = df_price.copy()
        
        # Feature Engineering
        ml_df['Returns'] = ml_df['Close'].pct_change()
        ml_df['SMA_10'] = ml_df['Close'].rolling(10).mean()
        ml_df['RSI'] = 100 - (100 / (1 + ml_df['Close'].diff().clip(lower=0).rolling(14).mean() / (-ml_df['Close'].diff().clip(upper=0).rolling(14).mean() + 1e-9)))
        ml_df['Target'] = (ml_df['Close'].shift(-1) > ml_df['Close']).astype(int) # Next day up/down
        
        ml_df = ml_df.dropna()
        
        if len(ml_df) < 50:
            st.warning("Not enough data points to train AI model. Try a longer date range.")
        else:
            features = ['Returns', 'RSI', 'Volume']
            X = ml_df[features]
            y = ml_df['Target']
            
            # Train/Test Split
            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            # Model Training
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluation
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            
            col_ai1, col_ai2 = st.columns(2)
            
            with col_ai1:
                st.metric("Model Accuracy (Test Set)", f"{acc*100:.2f}%")
                st.write("The model attempts to predict if the Close price will be HIGHER the next day.")
            
            with col_ai2:
                # Latest Prediction
                last_row = X.iloc[[-1]]
                next_pred = model.predict(last_row)[0]
                prob = model.predict_proba(last_row)[0][1]
                
                direction = "UP 🟢" if next_pred == 1 else "DOWN 🔴"
                st.metric("Next Day Prediction", direction, f"Prob: {prob:.2f}")

    # ------------------------------------------------------------------
    # TAB 6: SCREENER
    # ------------------------------------------------------------------
    with tabs[5]:
        st.subheader("⚡ Quick Compare Screener")
        
        default_tickers = "AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, AMD"
        universe = st.text_area("Enter tickers (comma separated)", value=default_tickers)
        
        if st.button("Run Screener"):
            ticker_list = [t.strip().upper() for t in universe.split(",") if t.strip()]
            
            results = []
            progress = st.progress(0)
            
            for i, t_sym in enumerate(ticker_list):
                try:
                    t_info = fetch_stock_info(t_sym)
                    if t_info:
                        results.append({
                            "Symbol": t_sym,
                            "Price": t_info.get('currentPrice', 0),
                            "Market Cap": t_info.get('marketCap', 0),
                            "P/E": t_info.get('trailingPE', 0),
                            "Beta": t_info.get('beta', 0),
                            "52W High": t_info.get('fiftyTwoWeekHigh', 0),
                            "Sector": t_info.get('sector', 'N/A')
                        })
                except:
                    pass
                progress.progress((i + 1) / len(ticker_list))
            
            if results:
                sdf = pd.DataFrame(results)
                
                # Apply formatting for display
                display_df = sdf.copy()
                display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: format_large_number(x))
                display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
                display_df['P/E'] = display_df['P/E'].apply(lambda x: f"{x:.2f}" if x else "-")
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("No data found for the provided tickers.")

if __name__ == "__main__":
    main()