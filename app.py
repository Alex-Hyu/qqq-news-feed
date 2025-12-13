import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
from datetime import timedelta
import pytz
import feedparser
from io import StringIO
from transformers import pipeline
from streamlit_autorefresh import st_autorefresh
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 0. 全局配置 ---
st.set_page_config(page_title="QQQ 宏观战情室 Pro Max", layout="wide", page_icon="🦅")

st.markdown("""
    <style>
    .metric-card {background-color: #f9f9f9; border-radius: 5px; padding: 10px; border: 1px solid #e0e0e0;}
    .news-card {padding: 10px; margin-bottom: 5px; border-radius: 5px; border-left: 5px solid #ccc;}
    .news-bull {background-color: #e6fffa; border-left-color: #00c04b;}
    .news-bear {background-color: #fff5f5; border-left-color: #ff4b4b;}
    .news-neutral {background-color: #f8f9fa; border-left-color: #6c757d;}
    .summary-box {padding: 15px; border-radius: 10px; margin-bottom: 20px;}
    .summary-bull {background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;}
    .summary-bear {background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;}
    .summary-neutral {background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db;}
    .calendar-urgent {background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 8px; margin: 4px 0; border-radius: 4px;}
    .calendar-soon {background-color: #e7f3ff; border-left: 4px solid #0d6efd; padding: 8px; margin: 4px 0; border-radius: 4px;}
    .calendar-normal {background-color: #f8f9fa; border-left: 4px solid #6c757d; padding: 8px; margin: 4px 0; border-radius: 4px;}
    .importance-5 {color: #dc3545; font-weight: bold;}
    .importance-4 {color: #fd7e14; font-weight: bold;}
    .importance-3 {color: #0d6efd;}
    .importance-2 {color: #6c757d;}
    .category-tag {display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; margin-right: 5px;}
    .tag-fed {background-color: #dc3545; color: white;}
    .tag-boj {background-color: #fd7e14; color: white;}
    .tag-ai {background-color: #6f42c1; color: white;}
    .tag-mag7 {background-color: #0d6efd; color: white;}
    .tag-crypto {background-color: #ffc107; color: black;}
    .tag-macro {background-color: #198754; color: white;}
    .regime-box {padding: 15px; border-radius: 10px; margin: 10px 0;}
    .regime-risk-on {background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border: 2px solid #28a745;}
    .regime-risk-off {background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); border: 2px solid #dc3545;}
    .regime-neutral {background: linear-gradient(135deg, #e2e3e5 0%, #d6d8db 100%); border: 2px solid #6c757d;}
    .export-box {background-color: #f0f7ff; border: 1px dashed #0d6efd; padding: 15px; border-radius: 8px; margin: 10px 0;}
    </style>
    """, unsafe_allow_html=True)

# --- [侧边栏] ---
with st.sidebar:
    st.header("⚙️ 设置")
    av_api_key = st.text_input("AlphaVantage API Key", value="UMWB63OXOOCIZHXR", type="password")
    st.divider()
    st.subheader("系统状态")
    count = st_autorefresh(interval=30 * 60 * 1000, key="data_refresher")
    st.caption(f"🟢 自动刷新: 开启 (30分钟)")
    if st.button("🔄 立即刷新"):
        st.rerun()

# ============================================================
# 1. 核心数据获取函数
# ============================================================

@st.cache_resource
def load_ai_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# --- SOFR/Repo 历史数据 (30天) ---
@st.cache_data(ttl=3600)
def get_sofr_repo_history():
    """获取 SOFR 和 Repo 利率的30天历史数据"""
    result = {
        'dates': [],
        'sofr': [],
        'tgcr': [],
        'spread': [],
        'current_sofr': 5.33,
        'current_tgcr': 5.32
    }
    try:
        # NY Fed API for historical rates
        end_date = datetime.date.today()
        start_date = end_date - timedelta(days=45)  # 多取一些确保有30个交易日
        
        url = f"https://markets.newyorkfed.org/api/rates/secured/sofr/search.json?startDate={start_date}&endDate={end_date}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            sofr_data = {}
            for item in data.get('refRates', []):
                date = item.get('effectiveDate', '')
                rate = item.get('percentRate', 0)
                sofr_data[date] = float(rate)
        
        # TGCR (Tri-Party General Collateral Rate)
        url_tgcr = f"https://markets.newyorkfed.org/api/rates/secured/tgcr/search.json?startDate={start_date}&endDate={end_date}"
        r2 = requests.get(url_tgcr, timeout=10)
        tgcr_data = {}
        if r2.status_code == 200:
            data2 = r2.json()
            for item in data2.get('refRates', []):
                date = item.get('effectiveDate', '')
                rate = item.get('percentRate', 0)
                tgcr_data[date] = float(rate)
        
        # 合并数据
        all_dates = sorted(set(sofr_data.keys()) & set(tgcr_data.keys()))[-30:]
        for date in all_dates:
            result['dates'].append(date)
            result['sofr'].append(sofr_data.get(date, 0))
            result['tgcr'].append(tgcr_data.get(date, 0))
            result['spread'].append(sofr_data.get(date, 0) - tgcr_data.get(date, 0))
        
        if result['sofr']:
            result['current_sofr'] = result['sofr'][-1]
            result['current_tgcr'] = result['tgcr'][-1]
            
    except Exception as e:
        st.warning(f"SOFR/Repo 历史数据获取失败: {e}")
    
    return result

# --- RRP/TGA 历史数据 (30天) ---
@st.cache_data(ttl=3600)
def get_rrp_tga_history():
    """获取 RRP 和 TGA 的30天历史数据"""
    result = {
        'dates': [],
        'rrp': [],
        'tga': [],
        'current_rrp': 0,
        'current_tga': 0,
        'rrp_chg': 0,
        'tga_chg': 0
    }
    try:
        # RRP (Overnight Reverse Repo)
        rrp_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=RRPONTSYD"
        rrp_df = pd.read_csv(rrp_url)
        rrp_df = rrp_df.dropna().tail(35)
        
        # TGA (Treasury General Account)
        tga_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WTREGEN"
        tga_df = pd.read_csv(tga_url)
        tga_df = tga_df.dropna().tail(35)
        
        # 对齐日期
        rrp_df['DATE'] = pd.to_datetime(rrp_df['DATE'])
        tga_df['DATE'] = pd.to_datetime(tga_df['DATE'])
        
        # 取最近30天
        result['dates'] = rrp_df['DATE'].dt.strftime('%Y-%m-%d').tolist()[-30:]
        result['rrp'] = rrp_df['RRPONTSYD'].tolist()[-30:]
        
        # TGA 是周度数据，需要对齐
        tga_dict = dict(zip(tga_df['DATE'].dt.strftime('%Y-%m-%d'), tga_df['WTREGEN']))
        result['tga'] = []
        last_tga = list(tga_dict.values())[-1] if tga_dict else 0
        for d in result['dates']:
            result['tga'].append(tga_dict.get(d, last_tga))
        
        if result['rrp']:
            result['current_rrp'] = result['rrp'][-1]
            result['rrp_chg'] = result['rrp'][-1] - result['rrp'][-2] if len(result['rrp']) > 1 else 0
        if result['tga']:
            result['current_tga'] = result['tga'][-1]
            result['tga_chg'] = result['tga'][-1] - result['tga'][-2] if len(result['tga']) > 1 else 0
            
    except Exception as e:
        st.warning(f"RRP/TGA 历史数据获取失败: {e}")
    
    return result

@st.cache_data(ttl=3600)
def get_ny_fed_data():
    try:
        url = "https://markets.newyorkfed.org/api/rates/all/latest.json"
        r = requests.get(url, timeout=5).json()
        rates = {'SOFR': 5.3, 'TGCR': 5.3} 
        for item in r.get('refRates', []):
            if item['type'] == 'SOFR': rates['SOFR'] = float(item['percentRate'])
            if item['type'] == 'TGCR': rates['TGCR'] = float(item['percentRate'])
        return rates
    except: return {'SOFR': 5.33, 'TGCR': 5.32}

@st.cache_data(ttl=3600)
def get_fed_liquidity():
    res = {"RRP": 0, "RRP_Chg": 0, "TGA": 0, "TGA_Chg": 0}
    try:
        rrp_df = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=RRPONTSYD")
        res['RRP'] = rrp_df.iloc[-1]['RRPONTSYD']
        res['RRP_Chg'] = res['RRP'] - rrp_df.iloc[-2]['RRPONTSYD']
        tga_df = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=WTREGEN")
        res['TGA'] = tga_df.iloc[-1]['WTREGEN']
        res['TGA_Chg'] = res['TGA'] - tga_df.iloc[-2]['WTREGEN']
    except: pass
    return res

@st.cache_data(ttl=1800)
def get_credit_spreads():
    try:
        data = yf.download(["HYG", "LQD"], period="5d", progress=False)['Close']
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        ratio = data['HYG'] / data['LQD']
        curr = ratio.iloc[-1]
        pct = ((curr - ratio.iloc[-2]) / ratio.iloc[-2]) * 100
        return curr, pct
    except: return 0, 0

@st.cache_data(ttl=1800)
def get_rates_and_fx():
    tickers = ["^IRX", "^TNX", "DX-Y.NYB", "JPY=X", "^MOVE"] 
    res = {'Yield_Short': 0, 'Yield_10Y': 0, 'Inversion': 0, 'DXY': 0, 'USDJPY': 0, 'MOVE': 0, 'USDJPY_Chg': 0}
    
    try:
        df = yf.download(tickers, period="1mo", group_by='ticker', progress=False)
        
        try:
            tnx_series = df['^TNX']['Close'].dropna()
            if not tnx_series.empty:
                res['Yield_10Y'] = tnx_series.iloc[-1]
        except: pass

        try:
            irx_series = df['^IRX']['Close'].dropna()
            if not irx_series.empty:
                res['Yield_Short'] = irx_series.iloc[-1]
        except: pass
        
        try:
            move_series = df['^MOVE']['Close']
            move_series = move_series.ffill().dropna()
            if not move_series.empty:
                res['MOVE'] = move_series.iloc[-1]
            else:
                res['MOVE'] = 0
        except: pass

        try:
            if not df['DX-Y.NYB']['Close'].dropna().empty: 
                res['DXY'] = df['DX-Y.NYB']['Close'].dropna().iloc[-1]
            jpy_series = df['JPY=X']['Close'].dropna()
            if not jpy_series.empty: 
                res['USDJPY'] = jpy_series.iloc[-1]
                if len(jpy_series) > 1:
                    res['USDJPY_Chg'] = jpy_series.iloc[-1] - jpy_series.iloc[-2]
        except: pass

        if res['Yield_10Y'] and res['Yield_Short']:
            res['Inversion'] = res['Yield_10Y'] - res['Yield_Short']

    except Exception as e:
        print(f"Rates Error: {e}")
        
    return res

@st.cache_data(ttl=1800)
def get_volatility_indices():
    data = {}
    try:
        vix = yf.Ticker("^VIX").history(period="2d")['Close'].iloc[-1]
        data['VIX'] = vix
    except: data['VIX'] = 15.0
    try:
        r = requests.get("https://api.alternative.me/fng/").json()
        data['Crypto_Val'] = int(r['data'][0]['value'])
        data['Crypto_Text'] = r['data'][0]['value_classification']
    except: data['Crypto_Val'] = 50; data['Crypto_Text'] = "Unknown"
    return data

@st.cache_data(ttl=1800)
def get_derivatives_structure():
    res = {
        "Futures_Basis": 0, "Basis_Status": "Normal", 
        "GEX_Net": "Neutral", "Call_Wall": 0, "Put_Wall": 0, 
        "Vanna_Status": "Neutral", "Current_Price": 0
    }
    try:
        market_data = yf.download(["NQ=F", "^NDX", "QQQ", "^VIX"], period="2d", progress=False)['Close']
        if isinstance(market_data.columns, pd.MultiIndex): market_data.columns = market_data.columns.get_level_values(0)
        
        fut = market_data['NQ=F'].iloc[-1]
        spot = market_data['^NDX'].iloc[-1]
        qqq_price = market_data['QQQ'].iloc[-1]
        res['Current_Price'] = qqq_price
        
        basis = fut - spot
        res['Futures_Basis'] = basis
        if basis < -15: res['Basis_Status'] = "🔴 Backwardation (极度看空)"
        elif basis > 60: res['Basis_Status'] = "🟢 Contango (正常)"
        else: res['Basis_Status'] = "⚪ Neutral"
        
        qqq = yf.Ticker("QQQ")
        expirations = qqq.options[:3] 
        all_calls = []; all_puts = []
        for date in expirations:
            try:
                chain = qqq.option_chain(date)
                c = chain.calls.fillna(0); p = chain.puts.fillna(0)
                c = c[c['openInterest'] > 100]; p = p[p['openInterest'] > 100]
                all_calls.append(c[['strike', 'openInterest']])
                all_puts.append(p[['strike', 'openInterest']])
            except: continue
        
        if all_calls:
            df_calls = pd.concat(all_calls).groupby('strike')['openInterest'].sum()
            df_puts = pd.concat(all_puts).groupby('strike')['openInterest'].sum()
            res['Call_Wall'] = df_calls.idxmax()
            res['Put_Wall'] = df_puts.idxmax()
            
            range_min = qqq_price * 0.98; range_max = qqq_price * 1.02
            calls_atm = df_calls[(df_calls.index >= range_min) & (df_calls.index <= range_max)].sum()
            puts_atm = df_puts[(df_puts.index >= range_min) & (df_puts.index <= range_max)].sum()
            gamma_ratio = puts_atm / max(1, calls_atm)
            
            if qqq_price < res['Put_Wall']: res['GEX_Net'] = "🔴 Negative Gamma (Crash Risk)"
            elif qqq_price > res['Call_Wall']: res['GEX_Net'] = "🟢 Positive Gamma (Breakout)"
            else:
                if gamma_ratio > 1.2: res['GEX_Net'] = "🟠 Weak Negative (震荡偏弱)"
                else: res['GEX_Net'] = "🟢 Positive Gamma (震荡偏强)"

        ndx_chg = spot - market_data['^NDX'].iloc[-2]
        vix_chg = market_data['^VIX'].iloc[-1] - market_data['^VIX'].iloc[-2]
        if ndx_chg > 0 and vix_chg < 0: res['Vanna_Status'] = "🟢 Tailwind (助涨)"
        elif ndx_chg < 0 and vix_chg > 0: res['Vanna_Status'] = "🔴 Headwind (助跌)"
    except: pass
    return res

@st.cache_data(ttl=1800)
def get_qqq_options_data():
    qqq = yf.Ticker("QQQ")
    res = {"PCR": 0.0, "Unusual": []}
    try:
        expirations = qqq.options[:3]
        total_c_vol = 0; total_p_vol = 0; unusual = []
        for date in expirations:
            try:
                chain = qqq.option_chain(date)
                calls = chain.calls.fillna(0); puts = chain.puts.fillna(0)
                total_c_vol += calls['volume'].sum(); total_p_vol += puts['volume'].sum()
                for opt_type, df, icon in [("CALL", calls, "🟢"), ("PUT", puts, "🔴")]:
                    hot = df[(df['volume'] > 2000) & (df['volume'] > df['openInterest'] * 1.2)]
                    for _, row in hot.iterrows():
                        unusual.append({
                            "Type": f"{icon} {opt_type}", "Strike": row['strike'], "Exp": date,
                            "Vol": int(row['volume']), "OI": int(row['openInterest']),
                            "Ratio": round(row['volume'] / (row['openInterest']+1), 1)
                        })
            except: continue
        if total_c_vol > 0: res['PCR'] = round(total_p_vol / total_c_vol, 2)
        res['Unusual'] = sorted(unusual, key=lambda x: x['Vol'], reverse=True)[:10]
    except: pass
    return res

@st.cache_data(ttl=60)
def get_intraday_tactics():
    res = {
        "VWAP": 0, "Price": 0, "Trend": "Neutral",
        "Exp_Move": 0, "Upper_Band": 0, "Lower_Band": 0,
        "0DTE_Call_Vol": 0, "0DTE_Put_Vol": 0, "0DTE_Sentiment": "Neutral",
        "Last_Update": datetime.datetime.now().strftime("%H:%M:%S")
    }
    try:
        df = yf.download("QQQ", period="1d", interval="1m", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['PV'] = df['TP'] * df['Volume']
            vwap = df['PV'].sum() / df['Volume'].sum() if df['Volume'].sum() > 0 else 0
            
            current_price = df['Close'].iloc[-1]
            res['VWAP'] = vwap
            res['Price'] = current_price
            
            if vwap > 0:
                if current_price > vwap * 1.001: res['Trend'] = "🟢 多头强势"
                elif current_price < vwap * 0.999: res['Trend'] = "🔴 空头压制"
                else: res['Trend'] = "⚪ 震荡"
            
        vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        exp_move = res['Price'] * ((vix/16)/100)
        res['Exp_Move'] = exp_move
        res['Upper_Band'] = res['Price'] + exp_move
        res['Lower_Band'] = res['Price'] - exp_move
        
        qqq = yf.Ticker("QQQ")
        target_date = qqq.options[0]
        chain = qqq.option_chain(target_date)
        c_vol = chain.calls['volume'].sum()
        p_vol = chain.puts['volume'].sum()
        res['0DTE_Call_Vol'] = c_vol
        res['0DTE_Put_Vol'] = p_vol
        
        ratio = p_vol / c_vol if c_vol > 0 else 1
        if ratio < 0.8: res['0DTE_Sentiment'] = "🟢 Call 主导"
        elif ratio > 1.2: res['0DTE_Sentiment'] = "🔴 Put 主导"
        else: res['0DTE_Sentiment'] = "⚪ 平衡"
        
        res['Expiry_Date'] = target_date
    except Exception as e: pass
    return res

# ============================================================
# 2. 新闻系统 (多源 + 重要性评分)
# ============================================================

# 关键词重要性权重
IMPORTANCE_WEIGHTS = {
    # 最高优先级 - 央行政策 (权重 5)
    "FOMC": 5, "rate decision": 5, "Powell": 5, "Ueda": 5, "Kuroda": 5,
    "rate cut": 5, "rate hike": 5, "QT": 5, "QE": 5, "tapering": 5,
    "Fed": 4, "BOJ": 4, "ECB": 4, "Bank of Japan": 4,
    
    # 高优先级 - 宏观数据 (权重 4)
    "CPI": 4, "inflation": 4, "employment": 4, "GDP": 4, "PCE": 4,
    "payroll": 4, "unemployment": 4, "Treasury": 4, "yield": 3,
    
    # 中高优先级 - 七巨头 (权重 3)
    "NVIDIA": 3, "NVDA": 3, "Apple": 3, "AAPL": 3, 
    "Microsoft": 3, "MSFT": 3, "Google": 3, "Alphabet": 3, "GOOGL": 3,
    "Amazon": 3, "AMZN": 3, "Meta": 3, "META": 3, 
    "Tesla": 3, "TSLA": 3,
    
    # 中优先级 - AI (权重 3)
    "OpenAI": 3, "ChatGPT": 3, "GPT-5": 4, "AI chip": 3, "GPU": 3,
    "artificial intelligence": 3, "machine learning": 2, "LLM": 3,
    "Anthropic": 3, "Claude": 2,
    
    # 加密货币 (权重 2)
    "Bitcoin": 2, "BTC": 2, "Ethereum": 2, "ETH": 2, "crypto": 2,
    
    # 一般财经 (权重 1-2)
    "earnings": 2, "revenue": 2, "guidance": 2,
    "stock": 1, "market": 1, "trading": 1
}

# 新闻分类
NEWS_CATEGORIES = {
    "fed": ["Fed", "FOMC", "Powell", "rate cut", "rate hike", "QT", "QE", "Treasury", "Federal Reserve"],
    "boj": ["BOJ", "Bank of Japan", "Ueda", "Kuroda", "yen", "JPY"],
    "ai": ["OpenAI", "ChatGPT", "GPT", "AI chip", "artificial intelligence", "LLM", "Anthropic", "Claude", "machine learning"],
    "mag7": ["NVIDIA", "NVDA", "Apple", "AAPL", "Microsoft", "MSFT", "Google", "Alphabet", "GOOGL", "Amazon", "AMZN", "Meta", "META", "Tesla", "TSLA"],
    "crypto": ["Bitcoin", "BTC", "Ethereum", "ETH", "crypto", "cryptocurrency"],
    "macro": ["CPI", "inflation", "GDP", "employment", "payroll", "unemployment", "PCE"]
}

def categorize_news(title: str) -> list:
    """对新闻进行分类"""
    categories = []
    title_lower = title.lower()
    for cat, keywords in NEWS_CATEGORIES.items():
        for kw in keywords:
            if kw.lower() in title_lower:
                categories.append(cat)
                break
    return list(set(categories)) if categories else ["general"]

def score_news_importance(title: str) -> int:
    """计算新闻重要性评分"""
    score = 0
    title_lower = title.lower()
    for keyword, weight in IMPORTANCE_WEIGHTS.items():
        if keyword.lower() in title_lower:
            score = max(score, weight)  # 取最高权重而非累加
    return score

@st.cache_data(ttl=1800)
def get_multi_source_news():
    """从多个来源获取新闻"""
    feeds = [
        # 宏观 & 美联储
        ("Fed", "https://www.federalreserve.gov/feeds/press_all.xml"),
        ("Reuters", "https://feeds.reuters.com/reuters/businessNews"),
        ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        
        # 科技 & AI
        ("TechCrunch", "https://techcrunch.com/feed/"),
        
        # 加密货币
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ]
    
    articles = []
    for src, url in feeds:
        try:
            f = feedparser.parse(url)
            for e in f.entries[:8]:  # 每个源取8条
                title = e.get('title', '')
                link = e.get('link', '')
                published = e.get('published', e.get('updated', ''))
                
                importance = score_news_importance(title)
                categories = categorize_news(title)
                
                articles.append({
                    "Title": title,
                    "Link": link,
                    "Source": src,
                    "Published": published,
                    "Importance": importance,
                    "Categories": categories
                })
        except Exception as e:
            continue
    
    # 按重要性排序
    articles = sorted(articles, key=lambda x: x['Importance'], reverse=True)
    
    return pd.DataFrame(articles)

# ============================================================
# 3. 宏观日历 (2025 关键日期 + 倒计时)
# ============================================================

MACRO_CALENDAR_2025 = [
    # FOMC 会议
    {"date": "2025-01-29", "event": "FOMC 利率决议", "type": "fed", "importance": 5},
    {"date": "2025-03-19", "event": "FOMC 利率决议 + 点阵图", "type": "fed", "importance": 5},
    {"date": "2025-05-07", "event": "FOMC 利率决议", "type": "fed", "importance": 5},
    {"date": "2025-06-18", "event": "FOMC 利率决议 + 点阵图", "type": "fed", "importance": 5},
    {"date": "2025-07-30", "event": "FOMC 利率决议", "type": "fed", "importance": 5},
    {"date": "2025-09-17", "event": "FOMC 利率决议 + 点阵图", "type": "fed", "importance": 5},
    {"date": "2025-11-05", "event": "FOMC 利率决议", "type": "fed", "importance": 5},
    {"date": "2025-12-17", "event": "FOMC 利率决议 + 点阵图", "type": "fed", "importance": 5},
    
    # CPI 数据 (通常在每月10-15日)
    {"date": "2025-01-15", "event": "CPI (12月)", "type": "inflation", "importance": 4},
    {"date": "2025-02-12", "event": "CPI (1月)", "type": "inflation", "importance": 4},
    {"date": "2025-03-12", "event": "CPI (2月)", "type": "inflation", "importance": 4},
    {"date": "2025-04-10", "event": "CPI (3月)", "type": "inflation", "importance": 4},
    {"date": "2025-05-13", "event": "CPI (4月)", "type": "inflation", "importance": 4},
    {"date": "2025-06-11", "event": "CPI (5月)", "type": "inflation", "importance": 4},
    {"date": "2025-07-11", "event": "CPI (6月)", "type": "inflation", "importance": 4},
    {"date": "2025-08-13", "event": "CPI (7月)", "type": "inflation", "importance": 4},
    {"date": "2025-09-10", "event": "CPI (8月)", "type": "inflation", "importance": 4},
    {"date": "2025-10-10", "event": "CPI (9月)", "type": "inflation", "importance": 4},
    {"date": "2025-11-13", "event": "CPI (10月)", "type": "inflation", "importance": 4},
    {"date": "2025-12-10", "event": "CPI (11月)", "type": "inflation", "importance": 4},
    
    # 非农就业 (通常在每月第一个周五)
    {"date": "2025-01-10", "event": "非农就业 (12月)", "type": "employment", "importance": 4},
    {"date": "2025-02-07", "event": "非农就业 (1月)", "type": "employment", "importance": 4},
    {"date": "2025-03-07", "event": "非农就业 (2月)", "type": "employment", "importance": 4},
    {"date": "2025-04-04", "event": "非农就业 (3月)", "type": "employment", "importance": 4},
    {"date": "2025-05-02", "event": "非农就业 (4月)", "type": "employment", "importance": 4},
    {"date": "2025-06-06", "event": "非农就业 (5月)", "type": "employment", "importance": 4},
    {"date": "2025-07-03", "event": "非农就业 (6月)", "type": "employment", "importance": 4},
    {"date": "2025-08-01", "event": "非农就业 (7月)", "type": "employment", "importance": 4},
    {"date": "2025-09-05", "event": "非农就业 (8月)", "type": "employment", "importance": 4},
    {"date": "2025-10-03", "event": "非农就业 (9月)", "type": "employment", "importance": 4},
    {"date": "2025-11-07", "event": "非农就业 (10月)", "type": "employment", "importance": 4},
    {"date": "2025-12-05", "event": "非农就业 (11月)", "type": "employment", "importance": 4},
    
    # BOJ 会议
    {"date": "2025-01-24", "event": "BOJ 利率决议", "type": "boj", "importance": 4},
    {"date": "2025-03-14", "event": "BOJ 利率决议", "type": "boj", "importance": 4},
    {"date": "2025-05-01", "event": "BOJ 利率决议", "type": "boj", "importance": 4},
    {"date": "2025-06-13", "event": "BOJ 利率决议", "type": "boj", "importance": 4},
    {"date": "2025-07-31", "event": "BOJ 利率决议", "type": "boj", "importance": 4},
    {"date": "2025-09-19", "event": "BOJ 利率决议", "type": "boj", "importance": 4},
    {"date": "2025-10-31", "event": "BOJ 利率决议", "type": "boj", "importance": 4},
    {"date": "2025-12-19", "event": "BOJ 利率决议", "type": "boj", "importance": 4},
    
    # 期权到期
    {"date": "2025-01-17", "event": "月度期权到期 (OPEX)", "type": "opex", "importance": 3},
    {"date": "2025-02-21", "event": "月度期权到期 (OPEX)", "type": "opex", "importance": 3},
    {"date": "2025-03-21", "event": "三巫日 (Quad Witching)", "type": "opex", "importance": 4},
    {"date": "2025-04-17", "event": "月度期权到期 (OPEX)", "type": "opex", "importance": 3},
    {"date": "2025-05-16", "event": "月度期权到期 (OPEX)", "type": "opex", "importance": 3},
    {"date": "2025-06-20", "event": "三巫日 (Quad Witching)", "type": "opex", "importance": 4},
    {"date": "2025-07-18", "event": "月度期权到期 (OPEX)", "type": "opex", "importance": 3},
    {"date": "2025-08-15", "event": "月度期权到期 (OPEX)", "type": "opex", "importance": 3},
    {"date": "2025-09-19", "event": "三巫日 (Quad Witching)", "type": "opex", "importance": 4},
    {"date": "2025-10-17", "event": "月度期权到期 (OPEX)", "type": "opex", "importance": 3},
    {"date": "2025-11-21", "event": "月度期权到期 (OPEX)", "type": "opex", "importance": 3},
    {"date": "2025-12-19", "event": "三巫日 (Quad Witching)", "type": "opex", "importance": 4},
]

def get_macro_calendar_with_countdown():
    """获取带倒计时的宏观日历"""
    today = datetime.date.today()
    upcoming = []
    
    for evt in MACRO_CALENDAR_2025:
        evt_date = datetime.datetime.strptime(evt["date"], "%Y-%m-%d").date()
        days_until = (evt_date - today).days
        
        if -1 <= days_until <= 60:  # 包括昨天到未来60天
            countdown = ""
            urgency = "normal"
            
            if days_until < 0:
                countdown = "昨天"
                urgency = "past"
            elif days_until == 0:
                countdown = "🔴 今天!"
                urgency = "urgent"
            elif days_until == 1:
                countdown = "🟠 明天"
                urgency = "urgent"
            elif days_until <= 3:
                countdown = f"⚠️ {days_until}天后"
                urgency = "soon"
            elif days_until <= 7:
                countdown = f"📅 {days_until}天后"
                urgency = "soon"
            else:
                countdown = f"{days_until}天后"
                urgency = "normal"
            
            upcoming.append({
                **evt,
                "countdown": countdown,
                "urgency": urgency,
                "days_until": days_until
            })
    
    return sorted(upcoming, key=lambda x: x["days_until"])

# ============================================================
# 4. Gamma 计算系统 (Black-Scholes)
# ============================================================

def black_scholes_gamma(S, K, T, r, sigma):
    """
    计算 Black-Scholes Gamma
    S: 现货价格
    K: 行权价
    T: 到期时间 (年)
    r: 无风险利率
    sigma: 隐含波动率
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0
    
    try:
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    except:
        return 0

@st.cache_data(ttl=1800)
def calculate_gex_profile():
    """
    计算完整的 GEX Profile
    返回按 Strike 分布的 Gamma Exposure
    """
    result = {
        'strikes': [],
        'gex_call': [],
        'gex_put': [],
        'gex_net': [],
        'total_gex': 0,
        'gamma_flip': 0,
        'max_pain': 0,
        'spot_price': 0,
        'put_wall': 0,
        'call_wall': 0
    }
    
    try:
        # 获取 QQQ 数据
        qqq = yf.Ticker("QQQ")
        spot = qqq.history(period="1d")['Close'].iloc[-1]
        result['spot_price'] = spot
        
        # 获取无风险利率 (3个月国债)
        try:
            irx = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100
        except:
            irx = 0.05
        
        # 收集所有期权链数据
        expirations = qqq.options[:4]  # 取前4个到期日
        all_options = []
        
        for exp_date in expirations:
            try:
                chain = qqq.option_chain(exp_date)
                
                # 计算到期时间
                exp_dt = datetime.datetime.strptime(exp_date, "%Y-%m-%d")
                today = datetime.datetime.now()
                T = max((exp_dt - today).days / 365, 0.001)
                
                # 处理 Calls
                for _, row in chain.calls.iterrows():
                    if row['openInterest'] > 50:
                        iv = row.get('impliedVolatility', 0.3)
                        if pd.isna(iv) or iv <= 0:
                            iv = 0.3
                        all_options.append({
                            'strike': row['strike'],
                            'oi': row['openInterest'],
                            'iv': iv,
                            'T': T,
                            'type': 'call'
                        })
                
                # 处理 Puts
                for _, row in chain.puts.iterrows():
                    if row['openInterest'] > 50:
                        iv = row.get('impliedVolatility', 0.3)
                        if pd.isna(iv) or iv <= 0:
                            iv = 0.3
                        all_options.append({
                            'strike': row['strike'],
                            'oi': row['openInterest'],
                            'iv': iv,
                            'T': T,
                            'type': 'put'
                        })
            except:
                continue
        
        if not all_options:
            return result
        
        # 计算每个 Strike 的 GEX
        gex_by_strike = {}
        
        for opt in all_options:
            strike = opt['strike']
            gamma = black_scholes_gamma(spot, strike, opt['T'], irx, opt['iv'])
            
            # GEX = Gamma × OI × 100 × Spot² / 1e9 (转换为十亿美元)
            gex = gamma * opt['oi'] * 100 * (spot ** 2) / 1e9
            
            if strike not in gex_by_strike:
                gex_by_strike[strike] = {'call': 0, 'put': 0}
            
            if opt['type'] == 'call':
                gex_by_strike[strike]['call'] += gex
            else:
                gex_by_strike[strike]['put'] += gex
        
        # 过滤并排序
        valid_strikes = [s for s in gex_by_strike.keys() if spot * 0.9 <= s <= spot * 1.1]
        valid_strikes = sorted(valid_strikes)
        
        for strike in valid_strikes:
            result['strikes'].append(strike)
            result['gex_call'].append(gex_by_strike[strike]['call'])
            result['gex_put'].append(-gex_by_strike[strike]['put'])  # Put GEX 为负
            result['gex_net'].append(gex_by_strike[strike]['call'] - gex_by_strike[strike]['put'])
        
        # 计算总 GEX
        result['total_gex'] = sum(result['gex_net'])
        
        # 找 Gamma Flip Point (净 GEX 从正变负的点)
        for i in range(len(result['strikes']) - 1):
            if result['gex_net'][i] > 0 and result['gex_net'][i+1] < 0:
                result['gamma_flip'] = (result['strikes'][i] + result['strikes'][i+1]) / 2
                break
            elif result['gex_net'][i] < 0 and result['gex_net'][i+1] > 0:
                result['gamma_flip'] = (result['strikes'][i] + result['strikes'][i+1]) / 2
                break
        
        # 找 Put Wall 和 Call Wall (最大 GEX 集中位置)
        if result['gex_call']:
            max_call_idx = result['gex_call'].index(max(result['gex_call']))
            result['call_wall'] = result['strikes'][max_call_idx]
        
        if result['gex_put']:
            max_put_idx = result['gex_put'].index(min(result['gex_put']))  # 最负的
            result['put_wall'] = result['strikes'][max_put_idx]
        
        # 计算 Max Pain
        # Max Pain = 期权到期时让最多期权价值归零的价格
        pain_by_strike = {}
        for opt in all_options:
            strike = opt['strike']
            if strike not in pain_by_strike:
                pain_by_strike[strike] = 0
            
            if opt['type'] == 'call':
                # Call 在 strike 以下全亏
                for s in valid_strikes:
                    if s < strike:
                        pain_by_strike[s] += opt['oi'] * 100 * (strike - s)
            else:
                # Put 在 strike 以上全亏
                for s in valid_strikes:
                    if s > strike:
                        pain_by_strike[s] += opt['oi'] * 100 * (s - strike)
        
        if pain_by_strike:
            result['max_pain'] = min(pain_by_strike, key=pain_by_strike.get)
        
    except Exception as e:
        st.warning(f"GEX 计算错误: {e}")
    
    return result

# ============================================================
# 5. 智能规则引擎
# ============================================================

def analyze_market_regime(ny_fed, fed_liq, credit, rates, vol, opt, deriv, rrp_tga_hist):
    """
    智能规则引擎：分析市场状态并生成信号
    """
    signals = []
    regime = "neutral"
    score = 0
    
    # ========== 流动性分析 ==========
    # 规则1: SOFR-Repo 利差
    spread = ny_fed['SOFR'] - ny_fed['TGCR']
    if spread > 0.10:
        signals.append({
            "level": "CRITICAL",
            "category": "流动性",
            "msg": f"🚨 银行间流动性紧缺: SOFR-Repo 利差 {spread:.3f}% 超过警戒线",
            "action": "减仓观望，关注 Fed 紧急操作",
            "score": -2
        })
        score -= 2
    elif spread > 0.05:
        signals.append({
            "level": "WARNING",
            "category": "流动性",
            "msg": f"⚠️ 流动性偏紧: SOFR-Repo 利差 {spread:.3f}%",
            "action": "谨慎持仓",
            "score": -1
        })
        score -= 1
    elif spread < 0.02:
        signals.append({
            "level": "POSITIVE",
            "category": "流动性",
            "msg": f"✅ 流动性充裕: SOFR-Repo 利差 {spread:.3f}%",
            "action": "环境有利于风险资产",
            "score": 1
        })
        score += 1
    
    # 规则2: RRP + TGA 联动
    rrp_chg = fed_liq['RRP_Chg']
    tga_chg = fed_liq['TGA_Chg']
    
    if rrp_chg < -50 and tga_chg > 30:
        signals.append({
            "level": "CRITICAL",
            "category": "流动性",
            "msg": f"🚨 双重抽水: RRP {rrp_chg:.0f}B 流出 + TGA 增加 {tga_chg:.0f}B",
            "action": "系统流动性急剧收缩，高度警惕",
            "score": -2
        })
        score -= 2
    elif rrp_chg > 50:
        signals.append({
            "level": "POSITIVE",
            "category": "流动性",
            "msg": f"✅ RRP 注水: {rrp_chg:.0f}B 流入隔夜逆回购",
            "action": "流动性改善",
            "score": 1
        })
        score += 1
    
    # 规则3: 信用利差
    if credit[1] < -1.0:
        signals.append({
            "level": "CRITICAL",
            "category": "风险偏好",
            "msg": f"🚨 信用风险飙升: HYG/LQD 单日下跌 {credit[1]:.2f}%",
            "action": "资金撤离垃圾债，Risk-Off 模式",
            "score": -2
        })
        score -= 2
    elif credit[1] < -0.3:
        signals.append({
            "level": "WARNING",
            "category": "风险偏好",
            "msg": f"⚠️ 信用偏紧: HYG/LQD 下跌 {credit[1]:.2f}%",
            "action": "关注信用市场动态",
            "score": -1
        })
        score -= 1
    elif credit[1] > 0.5:
        signals.append({
            "level": "POSITIVE",
            "category": "风险偏好",
            "msg": f"✅ 风险偏好回升: HYG/LQD 上涨 {credit[1]:.2f}%",
            "action": "Risk-On 环境",
            "score": 1
        })
        score += 1
    
    # ========== 美债分析 ==========
    # 规则4: 10Y 收益率
    if rates['Yield_10Y'] > 5.0:
        signals.append({
            "level": "CRITICAL",
            "category": "美债",
            "msg": f"🚨 利率风暴: 10Y 收益率 {rates['Yield_10Y']:.2f}% 突破 5%",
            "action": "科技股估值承压，减持高久期资产",
            "score": -2
        })
        score -= 2
    elif rates['Yield_10Y'] > 4.5:
        signals.append({
            "level": "WARNING",
            "category": "美债",
            "msg": f"⚠️ 利率压力: 10Y 收益率 {rates['Yield_10Y']:.2f}%",
            "action": "关注成长股表现",
            "score": -1
        })
        score -= 1
    
    # 规则5: MOVE 指数
    if rates['MOVE'] > 130:
        signals.append({
            "level": "CRITICAL",
            "category": "美债",
            "msg": f"🚨 债市恐慌: MOVE {rates['MOVE']:.0f} 极端波动",
            "action": "流动性危机风险，现金为王",
            "score": -2
        })
        score -= 2
    elif rates['MOVE'] > 110:
        signals.append({
            "level": "WARNING",
            "category": "美债",
            "msg": f"⚠️ 债市波动: MOVE {rates['MOVE']:.0f}",
            "action": "注意抵押品价值波动",
            "score": -1
        })
        score -= 1
    
    # 规则6: 收益率曲线
    if rates['Inversion'] < -1.0:
        signals.append({
            "level": "WARNING",
            "category": "美债",
            "msg": f"⚠️ 深度倒挂: 10Y-3M = {rates['Inversion']:.2f}%",
            "action": "经济衰退前瞻指标亮灯",
            "score": -0.5
        })
        score -= 0.5
    
    # ========== 日元套利分析 ==========
    # 规则7: USDJPY + VIX 联动
    if rates.get('USDJPY_Chg', 0) < -2 and vol['VIX'] > 20:
        signals.append({
            "level": "CRITICAL",
            "category": "汇率",
            "msg": f"🚨 日元套利平仓: USDJPY 急跌 + VIX {vol['VIX']:.1f}",
            "action": "全球 Risk-Off，减持风险资产",
            "score": -2
        })
        score -= 2
    
    # ========== 恐慌指标 ==========
    # 规则8: VIX
    if vol['VIX'] > 30:
        signals.append({
            "level": "CRITICAL",
            "category": "恐慌",
            "msg": f"🚨 VIX 恐慌: {vol['VIX']:.1f}",
            "action": "市场极度恐慌，可能是反弹机会",
            "score": -1  # 恐慌时反而可能见底
        })
        score -= 1
    elif vol['VIX'] > 25:
        signals.append({
            "level": "WARNING",
            "category": "恐慌",
            "msg": f"⚠️ VIX 升高: {vol['VIX']:.1f}",
            "action": "波动加剧，控制仓位",
            "score": -1
        })
        score -= 1
    elif vol['VIX'] < 12:
        signals.append({
            "level": "WARNING",
            "category": "恐慌",
            "msg": f"⚠️ VIX 过低: {vol['VIX']:.1f} (自满信号)",
            "action": "市场过于乐观，注意突发风险",
            "score": -0.5
        })
        score -= 0.5
    
    # 规则9: 币圈恐慌贪婪
    if vol['Crypto_Val'] < 20:
        signals.append({
            "level": "POSITIVE",
            "category": "情绪",
            "msg": f"✅ 币圈极度恐慌: {vol['Crypto_Val']} ({vol['Crypto_Text']})",
            "action": "反向指标，可能是买入时机",
            "score": 0.5
        })
        score += 0.5
    elif vol['Crypto_Val'] > 80:
        signals.append({
            "level": "WARNING",
            "category": "情绪",
            "msg": f"⚠️ 币圈极度贪婪: {vol['Crypto_Val']} ({vol['Crypto_Text']})",
            "action": "过热信号，谨慎追高",
            "score": -0.5
        })
        score -= 0.5
    
    # ========== 交易结构 ==========
    # 规则10: Gamma 环境
    if "Negative" in deriv['GEX_Net'] or "Crash" in deriv['GEX_Net']:
        signals.append({
            "level": "WARNING",
            "category": "结构",
            "msg": f"⚠️ 负 Gamma 环境: {deriv['GEX_Net']}",
            "action": "做市商追涨杀跌，波动放大",
            "score": -1
        })
        score -= 1
    elif "Positive" in deriv['GEX_Net']:
        signals.append({
            "level": "POSITIVE",
            "category": "结构",
            "msg": f"✅ 正 Gamma 环境: {deriv['GEX_Net']}",
            "action": "做市商高抛低吸，波动收敛",
            "score": 1
        })
        score += 1
    
    # 规则11: 期货基差
    if "Backwardation" in deriv['Basis_Status']:
        signals.append({
            "level": "CRITICAL",
            "category": "结构",
            "msg": f"🚨 期货贴水: 基差 {deriv['Futures_Basis']:.1f}",
            "action": "极度恐慌或强烈对冲需求",
            "score": -2
        })
        score -= 2
    
    # ========== 确定市场状态 ==========
    if score >= 3:
        regime = "risk_on"
    elif score <= -3:
        regime = "risk_off"
    else:
        regime = "neutral"
    
    return {
        "signals": signals,
        "regime": regime,
        "score": score
    }

# ============================================================
# 6. 宏观评分计算 (保留原有逻辑)
# ============================================================

def calculate_macro_score(ny_fed, fed_liq, credit, rates, vol, opt, deriv, news_score_val):
    score = 0; flags = []
    
    # 流动性
    liq_score = 0
    spread = ny_fed['SOFR'] - ny_fed['TGCR']
    if spread > 0.05: liq_score -= 1.0; flags.append("🔴 流动性紧缺 (SOFR > Repo)")
    elif spread < 0.02: liq_score += 0.5
    if fed_liq['RRP_Chg'] > 20: liq_score -= 0.5; flags.append("🔴 RRP 抽水")
    if fed_liq['TGA_Chg'] > 20: liq_score -= 0.5; flags.append("🔴 TGA 抽水")
    if credit[1] < -0.5: liq_score -= 0.5; flags.append("🔴 HYG/LQD 避险模式 (Credit Stress)")
    elif credit[1] > 0.2: liq_score += 0.5
    score += max(-2.5, min(2.5, liq_score))
    
    # 美债
    bond_score = 0
    if rates['Yield_10Y'] > 4.5: bond_score -= 1.0; flags.append("🔴 10Y 收益率过高 (>4.5%)")
    elif rates['Yield_10Y'] < 4.0: bond_score += 1.0
    if rates['MOVE'] > 110: bond_score -= 1.5; flags.append("🔴 MOVE 债市恐慌")
    if rates['Inversion'] < -0.5: flags.append("⚠️ 收益率深度倒挂 (Recession Risk)")
    score += max(-2.5, min(2.5, bond_score))
    
    # 恐慌
    fear_score = 0
    if vol['VIX'] > 25: fear_score -= 1.0; flags.append("🔴 VIX 恐慌模式")
    elif vol['VIX'] < 13: fear_score -= 0.5; flags.append("⚠️ VIX 过低 (自满)")
    if vol['Crypto_Val'] < 20: fear_score += 0.5; flags.append("🟢 币圈极度恐慌 (反向做多)")
    score += fear_score
    
    # 交易
    trade_score = 0
    if opt['PCR'] > 1.2: trade_score -= 0.5; flags.append("📉 PCR 极高 (空头拥挤)")
    elif opt['PCR'] < 0.6: trade_score += 0.5; flags.append("📈 PCR 极低 (多头拥挤)")
    if deriv['Basis_Status'].startswith("🔴"): trade_score -= 1.0; flags.append("🔴 期货贴水 (Hedging Demand)")
    if "Negative" in deriv['GEX_Net']: trade_score -= 0.5; flags.append("🔴 负 Gamma (高波动风险)")
    if "Headwind" in deriv['Vanna_Status']: flags.append("🔴 Vanna 阻力 (VIX Spike)")
    score += max(-2.0, min(2.0, trade_score))
    
    # 新闻
    score += news_score_val * 1.5
    
    final_score = round(score * (10 / 7.5), 1)
    summary = ""
    action = ""
    if final_score > 3:
        summary = "宏观环境 **偏多 (Bullish)**。流动性环境配合，市场情绪稳定。"
        action = "✅ **操作建议**: 逢低做多 (Buy Dips)，以 Call Wall 为目标位。"
    elif final_score < -3:
        summary = "宏观环境 **偏空 (Bearish)**。检测到流动性压力或市场恐慌指标异常。"
        action = "🛡️ **操作建议**: 现金为王，反弹做空 (Fade Rallies)，关注 Put Wall 支撑。"
    else:
        summary = "宏观环境 **中性震荡 (Neutral)**。多空信号交织，缺乏明确宏观驱动。"
        action = "⚖️ **操作建议**: 区间操作 (Range Trade)，避免追涨杀跌，以日内微观结构为主。"
    if not flags: flags.append("暂无显著异常指标")
    return final_score, flags, summary, action

# ============================================================
# 7. 生成导出到 Claude 的文本
# ============================================================

def generate_claude_export(ny_fed, fed_liq, credit, rates, vol, opt, deriv, gex_data, regime_analysis, processed_news):
    """生成可复制到 Claude 进行深度分析的文本"""
    
    export_text = f"""# 宏观战情室数据快照
生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EST

## 一、流动性指标
- SOFR: {ny_fed['SOFR']:.2f}%
- Repo (TGCR): {ny_fed['TGCR']:.2f}%
- SOFR-Repo 利差: {(ny_fed['SOFR'] - ny_fed['TGCR']):.3f}%
- RRP: ${fed_liq['RRP']:.0f}B (日变化: {fed_liq['RRP_Chg']:.0f}B)
- TGA: ${fed_liq['TGA']:.0f}B (日变化: {fed_liq['TGA_Chg']:.0f}B)
- HYG/LQD: {credit[0]:.3f} (日变化: {credit[1]:.2f}%)

## 二、美债与汇率
- 10Y 收益率: {rates['Yield_10Y']:.2f}%
- 3M 收益率: {rates['Yield_Short']:.2f}%
- 10Y-3M 利差: {rates['Inversion']:.2f}%
- MOVE 指数: {rates['MOVE']:.1f}
- DXY: {rates['DXY']:.2f}
- USDJPY: {rates['USDJPY']:.2f}

## 三、恐慌与情绪
- VIX: {vol['VIX']:.2f}
- 币圈恐慌贪婪: {vol['Crypto_Val']} ({vol['Crypto_Text']})
- PCR: {opt['PCR']:.2f}

## 四、交易微观结构
- 期货基差: {deriv['Futures_Basis']:.1f} ({deriv['Basis_Status']})
- Gamma 环境: {deriv['GEX_Net']}
- Vanna 状态: {deriv['Vanna_Status']}
- Put Wall: ${deriv['Put_Wall']:.0f}
- Call Wall: ${deriv['Call_Wall']:.0f}

## 五、GEX 分析
- 当前价格: ${gex_data['spot_price']:.2f}
- 净 GEX: {gex_data['total_gex']:.2f}B
- Gamma Flip Point: ${gex_data['gamma_flip']:.2f}
- Max Pain: ${gex_data['max_pain']:.2f}
- GEX Put Wall: ${gex_data['put_wall']:.2f}
- GEX Call Wall: ${gex_data['call_wall']:.2f}

## 六、规则引擎信号
市场状态: {regime_analysis['regime'].upper()}
综合评分: {regime_analysis['score']:.1f}

关键信号:
"""
    
    for sig in regime_analysis['signals']:
        export_text += f"- [{sig['level']}] {sig['msg']}\n"
    
    export_text += "\n## 七、重点新闻\n"
    for item in processed_news[:10]:
        cats = ", ".join(item.get('Categories', ['general']))
        export_text += f"- [{cats}] {item['Title']} (重要性: {item.get('Importance', 0)})\n"
    
    export_text += """
---
请基于以上数据进行深度分析:
1. 当前市场处于什么宏观周期？
2. 流动性环境对风险资产的影响？
3. 有哪些潜在的风险点？
4. 今日交易的最佳策略是什么？
"""
    
    return export_text

# ============================================================
# 8. 历史统计 (保留原有)
# ============================================================

@st.cache_data(ttl=86400)
def get_qqq_historical_stats():
    res = {}
    try:
        df = yf.download("QQQ", period="3y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df['Range'] = df['High'] - df['Low']
        df['Body'] = df['Close'] - df['Open']
        df['Abs_Body'] = df['Body'].abs()
        df['Efficiency'] = df['Abs_Body'] / df['Range']
        
        conditions = [
            (df['Efficiency'] > 0.5) & (df['Body'] > 0),
            (df['Efficiency'] > 0.5) & (df['Body'] < 0),
            (df['Efficiency'] <= 0.5)
        ]
        choices = ['Trend_Up', 'Trend_Down', 'Choppy']
        df['Type'] = np.select(conditions, choices, default='Choppy')
        
        counts = df['Type'].value_counts()
        total_days = len(df)
        
        res['Up_Days'] = counts.get('Trend_Up', 0)
        res['Down_Days'] = counts.get('Trend_Down', 0)
        res['Chop_Days'] = counts.get('Choppy', 0)
        
        res['Up_Pct'] = round((res['Up_Days'] / total_days) * 100, 1)
        res['Down_Pct'] = round((res['Down_Days'] / total_days) * 100, 1)
        res['Chop_Pct'] = round((res['Chop_Days'] / total_days) * 100, 1)
        
        res['Avg_Range'] = df['Range'].mean()
        res['Avg_Range_Pct'] = (df['Range'] / df['Open']).mean() * 100
        
    except Exception as e: 
        pass
    return res

# ============================================================
# UI 渲染
# ============================================================

# 数据加载
with st.spinner("正在聚合全市场数据..."):
    ai_model = load_ai_model()
    ny_fed = get_ny_fed_data()
    fed_liq = get_fed_liquidity()
    credit = get_credit_spreads()
    rates = get_rates_and_fx()
    vol = get_volatility_indices()
    opt = get_qqq_options_data()
    deriv = get_derivatives_structure()
    tactics = get_intraday_tactics()
    
    # 新增数据
    sofr_repo_hist = get_sofr_repo_history()
    rrp_tga_hist = get_rrp_tga_history()
    raw_news = get_multi_source_news()
    calendar_events = get_macro_calendar_with_countdown()
    gex_data = calculate_gex_profile()
    
    # 新闻情绪分析
    processed_news = []
    sentiment_total = 0
    weighted_sentiment = 0
    total_weight = 0
    
    if not raw_news.empty:
        for i, row in raw_news.head(15).iterrows():
            try:
                res = ai_model(row['Title'][:512])[0]
                label = res['label']
                score = res['score']
                sent = "Neutral"; val = 0
                if label == 'positive' and score > 0.5: sent="Bullish"; val=1
                elif label == 'negative' and score > 0.5: sent="Bearish"; val=-1
                
                importance = row.get('Importance', 1)
                weighted_sentiment += val * importance
                total_weight += importance
                sentiment_total += val
                
                processed_news.append({
                    **row.to_dict(), 
                    "Sentiment": sent,
                    "SentimentScore": val
                })
            except: pass
        
        avg_news_score = sentiment_total / max(1, len(processed_news))
        weighted_news_score = weighted_sentiment / max(1, total_weight)
    else: 
        avg_news_score = 0
        weighted_news_score = 0
    
    # 规则引擎分析
    regime_analysis = analyze_market_regime(ny_fed, fed_liq, credit, rates, vol, opt, deriv, rrp_tga_hist)
    
    # 原有评分系统
    final_score, flags, summary, action = calculate_macro_score(ny_fed, fed_liq, credit, rates, vol, opt, deriv, avg_news_score)

# ============================================================
# HEADER
# ============================================================
st.title("🦅 QQQ 宏观战情室 Pro Max")
current_time = datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M EST')
st.caption(f"Update: {current_time}")

# 战情综述
summary_class = "summary-bull" if final_score > 3 else "summary-bear" if final_score < -3 else "summary-neutral"
regime_text = "🟢 Risk-On" if regime_analysis['regime'] == 'risk_on' else "🔴 Risk-Off" if regime_analysis['regime'] == 'risk_off' else "⚪ Neutral"

st.markdown(f"""
<div class="summary-box {summary_class}">
    <h3>🛡️ 战情综述 (Score: {final_score}) | 市场状态: {regime_text}</h3>
    <p style="font-size:1.1em;">{summary}</p>
    <p><strong>🚨 异常指标监控 (Flags):</strong> { '  |  '.join(flags) }</p>
    <hr style="border-top: 1px dashed #ccc;">
    <p style="font-weight:bold;">{action}</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ============================================================
# 1. 流动性监控 (带图表)
# ============================================================
st.subheader("1. 💧 流动性监控")

l1, l2, l3, l4, l5 = st.columns(5)
l1.metric("SOFR", f"{ny_fed['SOFR']:.2f}%", f"Spread: {ny_fed['SOFR'] - ny_fed['TGCR']:.3f}")
l2.metric("Repo (TGCR)", f"{ny_fed['TGCR']:.2f}%")
l3.metric("RRP", f"${fed_liq['RRP']:.0f}B", f"{fed_liq['RRP_Chg']:.0f}B", delta_color="inverse")
l4.metric("TGA", f"${fed_liq['TGA']:.0f}B", f"{fed_liq['TGA_Chg']:.0f}B", delta_color="inverse")
l5.metric("HYG/LQD", f"{credit[0]:.3f}", f"{credit[1]:.2f}%", help="风险偏好指标")

with st.expander("📊 流动性历史图表 (30天)", expanded=True):
    tab1, tab2 = st.tabs(["SOFR / Repo 利差", "RRP / TGA"])
    
    with tab1:
        if sofr_repo_hist['dates']:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1,
                               subplot_titles=("SOFR vs Repo 利率", "SOFR-Repo 利差"))
            
            fig.add_trace(go.Scatter(x=sofr_repo_hist['dates'], y=sofr_repo_hist['sofr'],
                                    name='SOFR', line=dict(color='#0d6efd', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=sofr_repo_hist['dates'], y=sofr_repo_hist['tgcr'],
                                    name='TGCR (Repo)', line=dict(color='#198754', width=2)), row=1, col=1)
            
            spread_colors = ['#dc3545' if s > 0.05 else '#ffc107' if s > 0.02 else '#198754' for s in sofr_repo_hist['spread']]
            fig.add_trace(go.Bar(x=sofr_repo_hist['dates'], y=sofr_repo_hist['spread'],
                                name='利差', marker_color=spread_colors), row=2, col=1)
            fig.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="警戒线", row=2, col=1)
            
            fig.update_layout(height=400, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SOFR/Repo 历史数据暂不可用")
    
    with tab2:
        if rrp_tga_hist['dates']:
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.1,
                                subplot_titles=("RRP (隔夜逆回购)", "TGA (财政部账户)"))
            
            fig2.add_trace(go.Scatter(x=rrp_tga_hist['dates'], y=rrp_tga_hist['rrp'],
                                     name='RRP', fill='tozeroy', line=dict(color='#6f42c1', width=2)), row=1, col=1)
            fig2.add_trace(go.Scatter(x=rrp_tga_hist['dates'], y=rrp_tga_hist['tga'],
                                     name='TGA', fill='tozeroy', line=dict(color='#fd7e14', width=2)), row=2, col=1)
            
            fig2.update_layout(height=400, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("RRP/TGA 历史数据暂不可用")

st.divider()

# ============================================================
# 2. 美债与汇率
# ============================================================
st.subheader("2. 📈 美债与汇率")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("10Y 收益率", f"{rates['Yield_10Y']:.2f}%", help="全球资产定价之锚")
r2.metric("MOVE", f"{rates['MOVE']:.2f}", help="债市恐慌指数")
r3.metric("10Y/3M 倒挂", f"{rates['Inversion']:.2f}%", help="收益率曲线")
r4.metric("DXY", f"{rates['DXY']:.2f}")
r5.metric("USDJPY", f"{rates['USDJPY']:.2f}")

st.divider()

# ============================================================
# 3. 交易结构 + GEX Profile
# ============================================================
st.subheader("3. 🎯 交易与微观结构")

t1, t2, t3, t4 = st.columns(4)
t1.metric("PCR", f"{opt['PCR']}", help="Put/Call Ratio")
t2.metric("VIX", f"{vol['VIX']:.2f}")
t3.metric("币圈恐慌", f"{vol['Crypto_Val']}", f"{vol['Crypto_Text']}")
t4.metric("基差", f"{deriv['Futures_Basis']:.2f}", deriv['Basis_Status'])

g1, g2, g3, g4 = st.columns(4)
g1.metric("Gamma", deriv['GEX_Net'])
g2.metric("Vanna", deriv['Vanna_Status'])
g3.metric("Put Wall", f"${deriv['Put_Wall']}")
g4.metric("Call Wall", f"${deriv['Call_Wall']}")

with st.expander("📊 Gamma Exposure (GEX) Profile", expanded=True):
    if gex_data['strikes']:
        col_gex1, col_gex2 = st.columns([2, 1])
        
        with col_gex1:
            fig_gex = go.Figure()
            
            fig_gex.add_trace(go.Bar(
                x=gex_data['strikes'],
                y=gex_data['gex_call'],
                name='Call GEX',
                marker_color='#198754',
                opacity=0.7
            ))
            
            fig_gex.add_trace(go.Bar(
                x=gex_data['strikes'],
                y=gex_data['gex_put'],
                name='Put GEX',
                marker_color='#dc3545',
                opacity=0.7
            ))
            
            fig_gex.add_trace(go.Scatter(
                x=gex_data['strikes'],
                y=gex_data['gex_net'],
                name='Net GEX',
                line=dict(color='#0d6efd', width=3)
            ))
            
            fig_gex.add_vline(x=gex_data['spot_price'], line_dash="dash", line_color="yellow",
                             annotation_text=f"现价 ${gex_data['spot_price']:.2f}")
            
            if gex_data['gamma_flip'] > 0:
                fig_gex.add_vline(x=gex_data['gamma_flip'], line_dash="dot", line_color="orange",
                                 annotation_text=f"Gamma Flip ${gex_data['gamma_flip']:.0f}")
            
            fig_gex.update_layout(
                title="GEX Distribution by Strike",
                xaxis_title="Strike Price",
                yaxis_title="GEX (Billions $)",
                barmode='relative',
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig_gex, use_container_width=True)
        
        with col_gex2:
            st.markdown("#### 📍 关键位置")
            st.metric("当前价格", f"${gex_data['spot_price']:.2f}")
            st.metric("净 GEX", f"{gex_data['total_gex']:.2f}B", 
                     "正 Gamma ✅" if gex_data['total_gex'] > 0 else "负 Gamma ⚠️")
            st.metric("Gamma Flip", f"${gex_data['gamma_flip']:.2f}" if gex_data['gamma_flip'] > 0 else "N/A")
            st.metric("Max Pain", f"${gex_data['max_pain']:.2f}" if gex_data['max_pain'] > 0 else "N/A")
            st.metric("GEX Put Wall", f"${gex_data['put_wall']:.2f}" if gex_data['put_wall'] > 0 else "N/A")
            st.metric("GEX Call Wall", f"${gex_data['call_wall']:.2f}" if gex_data['call_wall'] > 0 else "N/A")
            
            st.markdown("---")
            st.markdown("**解读:**")
            if gex_data['total_gex'] > 0:
                st.success("正 Gamma: 做市商高抛低吸，波动收敛")
            else:
                st.warning("负 Gamma: 做市商追涨杀跌，波动放大")
    else:
        st.info("GEX 数据计算中...")

with st.expander("📚 战术手册：指标深度解读", expanded=False):
    st.markdown("""
    **1. HYG/LQD (信贷脉搏)**
    *   **定义**: 高收益债(Junk Bond)与投资级债(Corp Bond)的价格比率。
    *   **用法**: 它是股市的先行指标。如果 QQQ 在涨，但 HYG/LQD 在跌（背离），说明聪明的债券资金正在撤退。

    **2. MOVE 指数 (债市 VIX)**
    *   **定义**: 衡量美债收益率的波动率。
    *   **用法**: MOVE 是金融系统的"底层体温"。如果 MOVE 飙升 (>110)，意味着抵押品价值不稳定。

    **3. GEX (Gamma Exposure)**
    *   **定义**: 做市商持有的 Gamma 敞口总量。
    *   **用法**: 正 GEX = 低波动区间震荡; 负 GEX = 高波动单边行情。
    *   **Gamma Flip**: 正负 Gamma 翻转的价格点，是关键支撑/阻力。

    **4. Vanna & Charm**
    *   **Vanna**: VIX 下跌时，做市商买回对冲盘，助涨 (Vanna Rally)。
    *   **Charm**: 期权到期日前，价格被吸附在主力持仓区。
    """)

with st.expander("查看异动雷达", expanded=False):
    if opt['Unusual']: 
        st.dataframe(pd.DataFrame(opt['Unusual']), use_container_width=True)
    else: 
        st.info("无显著异动")

st.divider()

# ============================================================
# 4. 智能规则引擎分析
# ============================================================
st.subheader("4. 🧠 智能规则引擎分析")

signal_categories = {}
for sig in regime_analysis['signals']:
    cat = sig['category']
    if cat not in signal_categories:
        signal_categories[cat] = []
    signal_categories[cat].append(sig)

cols = st.columns(3)
col_idx = 0

for category, signals in signal_categories.items():
    with cols[col_idx % 3]:
        st.markdown(f"**{category}**")
        for sig in signals:
            if sig['level'] == 'CRITICAL':
                st.error(f"{sig['msg']}")
            elif sig['level'] == 'WARNING':
                st.warning(f"{sig['msg']}")
            elif sig['level'] == 'POSITIVE':
                st.success(f"{sig['msg']}")
            else:
                st.info(f"{sig['msg']}")
    col_idx += 1

st.markdown("---")
with st.expander("🤖 导出到 Claude 进行深度分析", expanded=False):
    export_text = generate_claude_export(ny_fed, fed_liq, credit, rates, vol, opt, deriv, gex_data, regime_analysis, processed_news)
    
    st.markdown("""
    <div class="export-box">
    <p>📋 点击下方按钮复制所有数据，然后粘贴到 Claude 进行深度分析</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.text_area("数据快照 (可复制)", export_text, height=400)
    st.caption("💡 提示: 全选文本框内容 (Ctrl+A)，复制 (Ctrl+C)，然后粘贴到 Claude 对话中")

st.divider()

# ============================================================
# 5. 宏观新闻 (多源 + 重要性排序)
# ============================================================
st.subheader("5. 📰 宏观新闻 (多源聚合)")

col_stat1, col_stat2, col_stat3 = st.columns(3)
col_stat1.metric("新闻情绪", f"{avg_news_score:.2f}", "Bullish" if avg_news_score > 0.2 else "Bearish" if avg_news_score < -0.2 else "Neutral")
col_stat2.metric("加权情绪", f"{weighted_news_score:.2f}", help="按重要性加权的情绪分数")
col_stat3.metric("新闻数量", len(processed_news))

def get_category_tag(cat):
    tag_map = {
        'fed': ('<span class="category-tag tag-fed">Fed</span>', '🏛️'),
        'boj': ('<span class="category-tag tag-boj">BOJ</span>', '🇯🇵'),
        'ai': ('<span class="category-tag tag-ai">AI</span>', '🤖'),
        'mag7': ('<span class="category-tag tag-mag7">七巨头</span>', '💎'),
        'crypto': ('<span class="category-tag tag-crypto">Crypto</span>', '₿'),
        'macro': ('<span class="category-tag tag-macro">Macro</span>', '📊'),
        'general': ('<span class="category-tag" style="background:#6c757d;color:white;">General</span>', '📰')
    }
    return tag_map.get(cat, tag_map['general'])

if processed_news:
    all_cats = set()
    for item in processed_news:
        all_cats.update(item.get('Categories', ['general']))
    
    selected_cat = st.selectbox("筛选分类", ["全部"] + sorted(list(all_cats)))
    
    for item in processed_news[:20]:
        cats = item.get('Categories', ['general'])
        
        if selected_cat != "全部" and selected_cat not in cats:
            continue
        
        importance = item.get('Importance', 0)
        sentiment = item.get('Sentiment', 'Neutral')
        
        if sentiment == "Bullish":
            css_class = "news-card news-bull"
        elif sentiment == "Bearish":
            css_class = "news-card news-bear"
        else:
            css_class = "news-card news-neutral"
        
        stars = "⭐" * min(importance, 5)
        
        cat_tags = ""
        for cat in cats:
            tag_html, _ = get_category_tag(cat)
            cat_tags += tag_html
        
        st.markdown(f"""
        <div class="{css_class}">
            <div>{cat_tags} {stars}</div>
            <strong>[{sentiment}]</strong> <a href="{item['Link']}" target="_blank">{item['Title']}</a>
            <br><small>来源: {item['Source']}</small>
        </div>
        """, unsafe_allow_html=True)
else:
    st.write("暂无新闻")

st.divider()

# ============================================================
# 6. 宏观日历 (带倒计时)
# ============================================================
st.subheader("6. 📅 宏观日历")

if calendar_events:
    urgent_events = [e for e in calendar_events if e['urgency'] == 'urgent']
    soon_events = [e for e in calendar_events if e['urgency'] == 'soon']
    normal_events = [e for e in calendar_events if e['urgency'] == 'normal']
    
    if urgent_events:
        st.markdown("### 🔴 紧急关注")
        for evt in urgent_events:
            st.markdown(f"""
            <div class="calendar-urgent">
                <strong>{evt['countdown']}</strong> | {evt['date']} | {evt['event']}
                <span style="float:right;">{'⭐' * evt['importance']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    if soon_events:
        st.markdown("### 🟠 近期事件")
        for evt in soon_events:
            st.markdown(f"""
            <div class="calendar-soon">
                <strong>{evt['countdown']}</strong> | {evt['date']} | {evt['event']}
                <span style="float:right;">{'⭐' * evt['importance']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander(f"📋 更多事件 ({len(normal_events)} 个)", expanded=False):
        for evt in normal_events:
            st.markdown(f"""
            <div class="calendar-normal">
                <strong>{evt['countdown']}</strong> | {evt['date']} | {evt['event']}
                <span style="float:right;">{'⭐' * evt['importance']}</span>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("暂无日历数据")

st.divider()

# ============================================================
# 7. 日内战术
# ============================================================
st.subheader("7. ⚔️ 日内战术面板")
st.caption(f"Snapshot: {tactics['Last_Update']}")

c_day1, c_day2, c_day3, c_day4 = st.columns(4)
c_day1.metric("VWAP", f"${tactics['VWAP']:.2f}", tactics['Trend'], delta_color="off")
c_day2.metric("预期波动", f"±${tactics['Exp_Move']:.2f}")
c_day3.metric("0DTE 情绪", tactics['0DTE_Sentiment'])

vwap_val = tactics['VWAP']
delta_str = "N/A"
if vwap_val > 0:
    pct = ((tactics['Price'] - vwap_val) / vwap_val) * 100
    delta_str = f"{pct:.2f}% vs VWAP"

c_day4.metric("QQQ 现价", f"${tactics['Price']:.2f}", delta_str)

with st.expander("🏹 日内指南", expanded=True):
    st.write(f"上轨: ${tactics['Upper_Band']:.2f} | 下轨: ${tactics['Lower_Band']:.2f}")
    st.write("策略: 价格 > VWAP 逢低多; 价格 < VWAP 逢高空。")

st.divider()

# ============================================================
# 8. 历史统计 (保留原有)
# ============================================================
st.subheader("8. 📊 策略回测：过去3年 QQQ K线形态统计")

with st.spinner("正在回测过去 3 年 K 线数据..."):
    stats = get_qqq_historical_stats()

if stats:
    c_stat1, c_stat2 = st.columns([1, 2])
    
    with c_stat1:
        st.markdown("#### 📊 市场性格画像")
        st.metric("震荡/均值回归概率", f"{stats['Chop_Pct']}%", f"{stats['Chop_Days']} 天", delta_color="off")
        st.metric("单边上涨概率", f"{stats['Up_Pct']}%", f"{stats['Up_Days']} 天", delta_color="normal")
        st.metric("单边下跌概率", f"{stats['Down_Pct']}%", f"{stats['Down_Days']} 天", delta_color="inverse")
        
        st.info(f"💡 **日内平均波幅 (ATR)**: ${stats['Avg_Range']:.2f} ({stats['Avg_Range_Pct']:.2f}%)")

    with c_stat2:
        st.markdown("#### 🧠 量化交易启示录")
        
        if stats['Chop_Pct'] > 50:
            strategy = "🛡️ **首选策略: 均值回归 (Mean Reversion)**"
            details = """
            *   **不要追涨杀跌**: 突破买入的胜率很低。
            *   **VWAP 战法**: 价格偏离 VWAP 过远时，大概率会回归。
            *   **期权**: 适合卖方策略 (Iron Condor) 或在关键支撑阻力位做反转。
            """
        else:
            strategy = "🚀 **首选策略: 趋势跟随 (Trend Following)**"
            details = """
            *   **顺势而为**: 突破关键点位后果断追单。
            *   **VWAP 战法**: 回踩 VWAP 不破是最佳上车点。
            *   **期权**: 买入 Call/Put 赌单边。
            """
            
        st.success(f"{strategy}")
        st.markdown(details)
        
        st.markdown("""
        ---
        **数据解读**:
        *   **震荡日 (Choppy)**: 收盘价回撤，留有长影线。适合 **高抛低吸**。
        *   **趋势日 (Trend)**: 收盘价在全天最高/最低附近。适合 **持有到收盘**。
        *   **统计结论**: 美股大部分时间 (约 60%+) 处于震荡中，单边暴跌或暴涨其实是少数。**日内交易切忌频繁止损去赌突破。**
        """)
