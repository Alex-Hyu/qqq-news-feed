import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline
import datetime
import feedparser
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="QQQ Pro Feed", layout="wide", page_icon="🦅")

# --- CACHED FUNCTIONS ---
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_data(ttl=300)
def get_market_data():
    """Fetches Prices AND News."""
    # 1. Get Live Prices
    tickers = ["QQQ", "^TNX", "^VIX"]
    price_data = {}
    try:
        data = yf.download(tickers, period="2d", progress=False)['Close']
        # Handle multi-index columns if necessary (yfinance update quirks)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0) # Flatten if needed
            
        for ticker in tickers:
            try:
                # Calculate % change
                prev = data[ticker].iloc[-2] # Yesterday close
                curr = data[ticker].iloc[-1] # Today current
                pct = ((curr - prev) / prev) * 100
                price_data[ticker] = pct
            except:
                price_data[ticker] = 0.0
    except:
        price_data = {"QQQ": 0.0, "^TNX": 0.0, "^VIX": 0.0}

    # 2. Get News (Yahoo + RSS)
    all_news = []
    
    # Yahoo News (Tech Focus)
    stock_tickers = ["QQQ", "NVDA", "AAPL", "MSFT"]
    for t in stock_tickers:
        try:
            stock = yf.Ticker(t)
            for item in stock.news:
                all_news.append({
                    "Title": item['title'],
                    "Link": item['link'],
                    "Source": item['publisher'],
                    "Time": datetime.datetime.fromtimestamp(item['providerPublishTime']),
                    "Type": "Stock"
                })
        except: pass

    # RSS (Macro)
    feeds = [
        ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("MarketWatch", "http://feeds.marketwatch.com/marketwatch/topstories/")
    ]
    for name, url in feeds:
        try:
            f = feedparser.parse(url)
            for e in f.entries[:5]:
                all_news.append({
                    "Title": e.title,
                    "Link": e.link,
                    "Source": name,
                    "Time": datetime.datetime.now(), # Approx
                    "Type": "Macro"
                })
        except: pass
        
    df_news = pd.DataFrame(all_news)
    if not df_news.empty:
        df_news = df_news.drop_duplicates(subset=['Title']).sort_values(by="Time", ascending=False)
        
    return price_data, df_news

def analyze_qqq_impact(headline, sentiment_label):
    h = headline.lower()
    # Specialized Logic
    if "fed" in h or "powell" in h:
        if "hike" in h or "caution" in h: return "Bearish (Fed Fear)"
        if "cut" in h or "dove" in h: return "Bullish (Fed Hope)"
    if "inflation" in h or "cpi" in h:
        if "rise" in h or "hot" in h: return "Bearish (Inflation)"
        if "fall" in h or "cool" in h: return "Bullish (Disinflation)"
    if "yield" in h and "jump" in h: return "Bearish (Rates)"
    
    # Fallback to AI
    if sentiment_label == "positive": return "Bullish"
    if sentiment_label == "negative": return "Bearish"
    return "Neutral"

# --- MAIN APP ---
st.title("🦅 QQQ Alpha-Seeker")

with st.spinner("Analyzing Market Conditions..."):
    prices, df_news = get_market_data()
    sentiment_pipe = load_sentiment_model()

# 1. DASHBOARD (THE REALITY CHECK)
st.subheader("Market Reality Check")
col1, col2, col3 = st.columns(3)

# QQQ Logic
q_val = prices.get("QQQ", 0)
q_color = "normal" if -0.5 < q_val < 0.5 else "inverse"
col1.metric("QQQ Price", f"{q_val:.2f}%", delta_color=q_color)

# 10Y Yield Logic (Inverted: Yield UP is BAD for Tech)
t_val = prices.get("^TNX", 0)
col2.metric("10Y Yield", f"{t_val:.2f}%", delta_color="inverse")

# VIX Logic (Inverted: Volatility UP is BAD)
v_val = prices.get("^VIX", 0)
col3.metric("VIX (Fear Index)", f"{v_val:.2f}%", delta_color="inverse")

st.divider()

# 2. NEWS ANALYSIS
results = []
if not df_news.empty:
    bar = st.progress(0, "AI Reading Headlines...")
    subset = df_news.head(15)
    
    for i, row in subset.iterrows():
        try:
            out = sentiment_pipe(row['Title'][:512])[0]
            impact = analyze_qqq_impact(row['Title'], out['label'])
            results.append({**row, "Signal": impact})
        except: pass
        bar.progress((i+1)/len(subset))
    
    bar.empty()
    df_res = pd.DataFrame(results)
    
    # Divergence Warning
    bull_count = len(df_res[df_res['Signal'].str.contains("Bullish")])
    bear_count = len(df_res[df_res['Signal'].str.contains("Bearish")])
    
    st.subheader("Sentiment vs. Price")
    
    # Logic: If News says Buy, but Price says Sell
    if bull_count > bear_count and q_val < -0.3:
        st.error(f"🚨 DIVERGENCE ALERT: News is Bullish ({bull_count} articles), but QQQ is Red ({q_val:.2f}%). Beware of 'Bull Trap'.")
    elif bear_count > bull_count and q_val > 0.3:
        st.success(f"🚀 STRENGTH ALERT: News is Bearish ({bear_count} articles), but QQQ is Green ({q_val:.2f}%). Market is ignoring bad news (Very Bullish).")
    else:
        st.info("Sentiment aligns with Price Action.")

    # Display Feed
    for i, row in df_res.iterrows():
        icon = "🟢" if "Bullish" in row['Signal'] else "🔴" if "Bearish" in row['Signal'] else "⚪"
        with st.expander(f"{icon} {row['Signal']} | {row['Title']}"):
            st.caption(f"Source: {row['Source']}")
            st.markdown(f"[Read Article]({row['Link']})")
