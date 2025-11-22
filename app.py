import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="QQQ Market Pulse", layout="wide", page_icon="📈")

# --- CACHED FUNCTIONS (Speed up performance) ---
@st.cache_resource
def load_sentiment_model():
    """Loads the FinBERT model once to avoid reloading on every click."""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_news():
    """Fetches news for QQQ and major Macro indicators from Yahoo Finance."""
    tickers = ["QQQ", "SPY", "^TNX", "NVDA", "AAPL", "MSFT"] # Tech & Macro focus
    all_news = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            for item in news:
                all_news.append({
                    "Title": item['title'],
                    "Link": item['link'],
                    "Publisher": item['publisher'],
                    "Time": datetime.datetime.fromtimestamp(item['providerPublishTime']),
                    "Related": ticker
                })
        except Exception:
            continue
            
    # Remove duplicates based on Title
    df = pd.DataFrame(all_news)
    if not df.empty:
        df = df.drop_duplicates(subset=['Title']).sort_values(by="Time", ascending=False)
    return df

def analyze_qqq_impact(headline, sentiment_label, sentiment_score):
    """
    Applies specific logic for the Nasdaq-100 (Tech).
    """
    headline_lower = headline.lower()
    
    # 1. YIELD / RATES LOGIC (The enemy of QQQ)
    if "yield" in headline_lower or "treasury" in headline_lower or "rate" in headline_lower:
        if "high" in headline_lower or "jump" in headline_lower or "rise" in headline_lower:
            return "Bearish (Rates Up)"
        elif "fall" in headline_lower or "drop" in headline_lower or "cut" in headline_lower:
            return "Bullish (Rates Down)"

    # 2. INFLATION LOGIC
    if "cpi" in headline_lower or "inflation" in headline_lower:
        if "hot" in headline_lower or "rise" in headline_lower:
            return "Bearish (Inflation Risk)"
        if "cool" in headline_lower or "fall" in headline_lower:
            return "Bullish (Fed Pivot Hope)"

    # 3. DEFAULT SENTIMENT MAPPING
    # If sentiment is Positive, usually good for QQQ, unless it's "Economy too hot" logic
    if sentiment_label == "positive":
        return "Bullish"
    elif sentiment_label == "negative":
        return "Bearish"
    else:
        return "Neutral"

# --- MAIN APP UI ---
st.title("🦅 QQQ Macro & News Feed")
st.markdown("Aggregating news from **Nasdaq-100, 10Y Yields, and Fed Policy**.")

# Load Model (Show spinner first time)
with st.spinner("Loading AI Models..."):
    sentiment_pipe = load_sentiment_model()

# Fetch Data
with st.spinner("Fetching latest market news..."):
    df_news = get_news()

if df_news.empty:
    st.warning("No news found at the moment. Markets might be quiet.")
else:
    # Run Analysis
    results = []
    progress_bar = st.progress(0)
    total = len(df_news)
    
    # Limit to top 15 latest articles to save speed
    df_process = df_news.head(15)
    
    for i, row in df_process.iterrows():
        # AI Sentiment Analysis
        # We take the first 512 chars to fit BERT limits
        output = sentiment_pipe(row['Title'][:512])[0]
        label = output['label']
        score = output['score']
        
        # Apply QQQ Specific Logic
        qqq_signal = analyze_qqq_impact(row['Title'], label, score)
        
        results.append({
            "Time": row['Time'].strftime("%H:%M"),
            "Headline": row['Title'],
            "Source": row['Publisher'],
            "Link": row['Link'],
            "AI Sentiment": label,
            "QQQ Impact": qqq_signal
        })
        progress_bar.progress((i + 1) / len(df_process))
    
    progress_bar.empty()
    df_results = pd.DataFrame(results)

    # --- DASHBOARD HEADER ---
    st.divider()
    
    # Score Calculation
    bulls = len(df_results[df_results['QQQ Impact'].str.contains("Bullish")])
    bears = len(df_results[df_results['QQQ Impact'].str.contains("Bearish")])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Bullish Signals", bulls)
    col2.metric("Bearish Signals", bears)
    
    sentiment_diff = bulls - bears
    if sentiment_diff > 2:
        col3.success("Overall Outlook: **BULLISH**")
    elif sentiment_diff < -2:
        col3.error("Overall Outlook: **BEARISH**")
    else:
        col3.info("Overall Outlook: **NEUTRAL / MIXED**")

    st.divider()

    # --- NEWS FEED DISPLAY ---
    st.subheader("Latest Analysis")
    
    for idx, row in df_results.iterrows():
        with st.expander(f"{row['QQQ Impact']} | {row['Headline']}"):
            st.write(f"**Source:** {row['Source']} at {row['Time']}")
            st.write(f"**Base AI Sentiment:** {row['AI Sentiment']}")
            st.markdown(f"[Read Article]({row['Link']})")