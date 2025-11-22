import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline
import datetime
import feedparser
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="QQQ Super Feed", layout="wide", page_icon="🦅")

# --- CACHED FUNCTIONS ---
@st.cache_resource
def load_sentiment_model():
    """Loads the FinBERT model once."""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_data(ttl=300)
def get_news():
    """Fetches news from Yahoo Finance (Stocks) AND RSS Feeds (Macro Economy)."""
    all_news = []

    # SOURCE 1: Yahoo Finance (Targeted Tech Stocks)
    tickers = ["QQQ", "NVDA", "AAPL", "MSFT", "^TNX"] # ^TNX is 10-Year Yield
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            for item in news:
                publish_time = datetime.datetime.fromtimestamp(item['providerPublishTime'])
                all_news.append({
                    "Title": item['title'],
                    "Link": item['link'],
                    "Source": item['publisher'],
                    "Time": publish_time,
                    "Type": "Stock Specific"
                })
        except Exception:
            continue

    # SOURCE 2: RSS Feeds (Macro Economy)
    rss_feeds = [
        ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("MarketWatch Top", "http://feeds.marketwatch.com/marketwatch/topstories/")
    ]

    for source_name, url in rss_feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]: # Get top 5 from each
                # Parse time (struct_time to datetime)
                if hasattr(entry, 'published_parsed'):
                    dt = datetime.datetime.fromtimestamp(time.mktime(entry.published_parsed))
                else:
                    dt = datetime.datetime.now()

                all_news.append({
                    "Title": entry.title,
                    "Link": entry.link,
                    "Source": source_name,
                    "Time": dt,
                    "Type": "Macro Economy"
                })
        except Exception:
            continue

    # Deduplicate and Sort
    df = pd.DataFrame(all_news)
    if not df.empty:
        df = df.drop_duplicates(subset=['Title'])
        df = df.sort_values(by="Time", ascending=False)
    return df

def analyze_qqq_impact(headline, sentiment_label):
    """Applies QQQ Logic."""
    headline_lower = headline.lower()
    
    # Macro Rules
    if "fed" in headline_lower or "rates" in headline_lower:
        if "hike" in headline_lower or "raise" in headline_lower: return "Bearish (Rates)"
        if "cut" in headline_lower or "pivot" in headline_lower: return "Bullish (Rates)"
    
    if "inflation" in headline_lower or "cpi" in headline_lower:
        if "hot" in headline_lower or "rise" in headline_lower: return "Bearish (Inflation)"
        if "cool" in headline_lower or "fall" in headline_lower: return "Bullish (Inflation)"

    # Default AI
    if sentiment_label == "positive": return "Bullish"
    if sentiment_label == "negative": return "Bearish"
    return "Neutral"

# --- MAIN APP UI ---
st.title("🦅 QQQ Super Feed")
st.markdown("Tracking **Nasdaq-100**, **Fed Policy**, and **Global Economy**.")

# Load AI
with st.spinner("Waking up AI..."):
    sentiment_pipe = load_sentiment_model()

# Refresh Button
if st.button("Refresh News"):
    st.cache_data.clear()

# Fetch Data
with st.spinner("Scanning market news sources..."):
    df_news = get_news()

if df_news.empty:
    st.warning("No news found right now.")
else:
    results = []
    # Analyze top 20 articles
    df_process = df_news.head(20)
    
    # Progress bar for AI processing
    progress_text = "AI is reading headlines..."
    my_bar = st.progress(0, text=progress_text)

    for i, row in df_process.iterrows():
        # AI Analysis
        try:
            output = sentiment_pipe(row['Title'][:512])[0]
            label = output['label']
            impact = analyze_qqq_impact(row['Title'], label)
            
            results.append({
                "Headline": row['Title'],
                "Source": row['Source'],
                "Link": row['Link'],
                "QQQ Signal": impact,
                "Type": row['Type']
            })
        except Exception:
            pass
        my_bar.progress((i + 1) / len(df_process), text=progress_text)

    my_bar.empty()
    df_results = pd.DataFrame(results)

    # METRICS
    st.divider()
    bulls = len(df_results[df_results['QQQ Signal'].str.contains("Bullish")])
    bears = len(df_results[df_results['QQQ Signal'].str.contains("Bearish")])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Bullish News", bulls)
    c2.metric("Bearish News", bears)
    
    if bulls > bears:
        c3.success("Market Mood: **RISK ON**")
    elif bears > bulls:
        c3.error("Market Mood: **RISK OFF**")
    else:
        c3.info("Market Mood: **NEUTRAL**")
        
    st.divider()

    # TABS FOR VIEWING
    tab1, tab2 = st.tabs(["All News", "Macro Only"])
    
    with tab1:
        for idx, row in df_results.iterrows():
            color = "green" if "Bullish" in row['QQQ Signal'] else "red" if "Bearish" in row['QQQ Signal'] else "gray"
            with st.expander(f":{color}[{row['QQQ Signal']}] | {row['Headline']}"):
                st.write(f"**Source:** {row['Source']} ({row['Type']})")
                st.markdown(f"[Read Story]({row['Link']})")

    with tab2:
        macro_df = df_results[df_results['Type'] == "Macro Economy"]
        for idx, row in macro_df.iterrows():
             with st.expander(f"{row['QQQ Signal']} | {row['Headline']}"):
                st.write(f"**Source:** {row['Source']}")
                st.markdown(f"[Read Story]({row['Link']})")
