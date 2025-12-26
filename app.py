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
st.set_page_config(page_title="宏观战情观察室", layout="wide", page_icon="🦅")

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
    st.subheader("🔄 刷新控制")
    
    # 全局刷新 (所有数据)
    if st.button("🔄 全局刷新 (所有数据)", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.caption("⚠️ 全局刷新较慢，建议仅在开盘前使用")
    
    st.divider()
    
    # 分类刷新
    st.markdown("**按需刷新：**")
    
    col_ref1, col_ref2 = st.columns(2)
    with col_ref1:
        if st.button("📊 GEX/期权", use_container_width=True, help="刷新期权链和GEX计算"):
            # 清除期权相关缓存
            calculate_gex_profile.clear()
            get_qqq_options_data.clear()
            get_derivatives_structure.clear()
            st.rerun()
    
    with col_ref2:
        if st.button("📈 日内数据", use_container_width=True, help="刷新VWAP和盘中数据"):
            get_intraday_tactics.clear()
            st.rerun()
    
    col_ref3, col_ref4 = st.columns(2)
    with col_ref3:
        if st.button("📰 新闻", use_container_width=True, help="刷新新闻和情绪分析"):
            get_multi_source_news.clear()
            st.rerun()
    
    with col_ref4:
        if st.button("💧 流动性", use_container_width=True, help="刷新SOFR/RRP/TGA"):
            get_sofr_repo_history.clear()
            get_rrp_tga_history.clear()
            get_ny_fed_data.clear()
            get_fed_liquidity.clear()
            st.rerun()
    
    st.divider()
    st.subheader("📋 缓存策略")
    st.caption("""
    • 流动性/宏观: 4小时  
    • 美债/汇率: 2小时  
    • 新闻: 2小时  
    • 期权/GEX: 1小时  
    • 日内VWAP: 5分钟
    """)

# ============================================================
# 1. 核心数据获取函数
# ============================================================

@st.cache_resource
def load_ai_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# --- SOFR/Repo 历史数据 (30天) ---
@st.cache_data(ttl=14400)  # 4小时缓存 (宏观数据变化慢)
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
@st.cache_data(ttl=14400)  # 4小时缓存 (每天更新一次的数据)
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
        
        # 自动检测日期列名 (可能是 'DATE' 或 'date' 或第一列)
        date_col = None
        for col in rrp_df.columns:
            if col.upper() == 'DATE' or 'date' in col.lower():
                date_col = col
                break
        if date_col is None:
            date_col = rrp_df.columns[0]  # 使用第一列作为日期
        
        # 数据列
        rrp_col = 'RRPONTSYD' if 'RRPONTSYD' in rrp_df.columns else rrp_df.columns[1]
        
        rrp_df = rrp_df.dropna().tail(35)
        rrp_df[date_col] = pd.to_datetime(rrp_df[date_col])
        
        # TGA (Treasury General Account)
        tga_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WTREGEN"
        tga_df = pd.read_csv(tga_url)
        
        # 自动检测 TGA 列名
        tga_date_col = None
        for col in tga_df.columns:
            if col.upper() == 'DATE' or 'date' in col.lower():
                tga_date_col = col
                break
        if tga_date_col is None:
            tga_date_col = tga_df.columns[0]
        
        tga_col = 'WTREGEN' if 'WTREGEN' in tga_df.columns else tga_df.columns[1]
        
        tga_df = tga_df.dropna().tail(35)
        tga_df[tga_date_col] = pd.to_datetime(tga_df[tga_date_col])
        
        # 取最近30天
        result['dates'] = rrp_df[date_col].dt.strftime('%Y-%m-%d').tolist()[-30:]
        result['rrp'] = rrp_df[rrp_col].tolist()[-30:]
        
        # TGA 是周度数据，需要对齐 (使用前向填充)
        tga_dict = dict(zip(tga_df[tga_date_col].dt.strftime('%Y-%m-%d'), tga_df[tga_col]))
        result['tga'] = []
        last_tga = list(tga_dict.values())[-1] if tga_dict else 0
        for d in result['dates']:
            if d in tga_dict:
                last_tga = tga_dict[d]
            result['tga'].append(last_tga)
        
        if result['rrp']:
            result['current_rrp'] = result['rrp'][-1]
            result['rrp_chg'] = result['rrp'][-1] - result['rrp'][-2] if len(result['rrp']) > 1 else 0
        if result['tga']:
            result['current_tga'] = result['tga'][-1]
            result['tga_chg'] = result['tga'][-1] - result['tga'][-2] if len(result['tga']) > 1 else 0
            
    except Exception as e:
        # 静默失败，不显示警告，返回空结果
        pass
    
    return result

@st.cache_data(ttl=14400)  # 4小时缓存
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

@st.cache_data(ttl=14400)  # 4小时缓存
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

@st.cache_data(ttl=7200)  # 2小时缓存
def get_credit_spreads():
    try:
        data = yf.download(["HYG", "LQD"], period="5d", progress=False)['Close']
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        ratio = data['HYG'] / data['LQD']
        curr = ratio.iloc[-1]
        pct = ((curr - ratio.iloc[-2]) / ratio.iloc[-2]) * 100
        return curr, pct
    except: return 0, 0

@st.cache_data(ttl=7200)  # 2小时缓存
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

@st.cache_data(ttl=3600)  # 1小时缓存
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

@st.cache_data(ttl=3600)  # 1小时缓存
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

@st.cache_data(ttl=3600)  # 1小时缓存
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

@st.cache_data(ttl=300)  # 5分钟缓存 (日内数据需要较新)
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

@st.cache_data(ttl=7200)  # 2小时缓存 (新闻 + FinBERT 分析较慢)
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

@st.cache_data(ttl=3600)  # 1小时缓存 (可手动刷新获取最新)
def calculate_gex_profile():
    """
    计算完整的 GEX Profile
    返回按 Strike 分布的 Gamma Exposure
    """
    # 记录计算时间
    calc_time = datetime.datetime.now(pytz.timezone('US/Eastern'))
    
    # 计算 OI 数据日期 (前一个交易日)
    # 简化处理：如果是周一，OI 是周五的；否则是昨天的
    today = datetime.date.today()
    if today.weekday() == 0:  # 周一
        oi_date = today - timedelta(days=3)  # 周五
    elif today.weekday() == 6:  # 周日
        oi_date = today - timedelta(days=2)  # 周五
    elif today.weekday() == 5:  # 周六
        oi_date = today - timedelta(days=1)  # 周五
    else:
        oi_date = today - timedelta(days=1)  # 昨天
    
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
        'call_wall': 0,
        'calc_time': calc_time.strftime('%Y-%m-%d %H:%M:%S EST'),
        'oi_date': oi_date.strftime('%Y-%m-%d'),
        'oi_weekday': ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][oi_date.weekday()]
    }
    
    try:
        # 获取 QQQ 数据
        qqq = yf.Ticker("QQQ")
        hist = qqq.history(period="1d")
        if hist.empty:
            return result
        spot = float(hist['Close'].iloc[-1])
        result['spot_price'] = spot
        
        # 获取无风险利率 (3个月国债)
        try:
            irx_hist = yf.Ticker("^IRX").history(period="1d")
            if not irx_hist.empty:
                irx = float(irx_hist['Close'].iloc[-1]) / 100
            else:
                irx = 0.05
        except:
            irx = 0.05
        
        # 收集所有期权链数据
        try:
            expirations = qqq.options[:4]  # 取前4个到期日
        except:
            return result
            
        all_options = []
        
        for exp_date in expirations:
            try:
                chain = qqq.option_chain(exp_date)
                
                # 计算到期时间
                exp_dt = datetime.datetime.strptime(exp_date, "%Y-%m-%d")
                today = datetime.datetime.now()
                days_to_exp = (exp_dt - today).days
                T = max(days_to_exp / 365, 0.001)
                
                # 处理 Calls
                for _, row in chain.calls.iterrows():
                    try:
                        oi = row.get('openInterest', 0)
                        if pd.isna(oi):
                            oi = 0
                        oi = float(oi)
                        
                        if oi > 50:
                            iv = row.get('impliedVolatility', 0.3)
                            if pd.isna(iv) or iv <= 0 or iv > 5:  # IV 超过 500% 视为异常
                                iv = 0.3
                            iv = float(iv)
                            
                            strike = float(row['strike'])
                            all_options.append({
                                'strike': strike,
                                'oi': oi,
                                'iv': iv,
                                'T': T,
                                'type': 'call'
                            })
                    except:
                        continue
                
                # 处理 Puts
                for _, row in chain.puts.iterrows():
                    try:
                        oi = row.get('openInterest', 0)
                        if pd.isna(oi):
                            oi = 0
                        oi = float(oi)
                        
                        if oi > 50:
                            iv = row.get('impliedVolatility', 0.3)
                            if pd.isna(iv) or iv <= 0 or iv > 5:
                                iv = 0.3
                            iv = float(iv)
                            
                            strike = float(row['strike'])
                            all_options.append({
                                'strike': strike,
                                'oi': oi,
                                'iv': iv,
                                'T': T,
                                'type': 'put'
                            })
                    except:
                        continue
            except:
                continue
        
        if not all_options:
            return result
        
        # 计算每个 Strike 的 GEX
        gex_by_strike = {}
        
        for opt in all_options:
            try:
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
            except:
                continue
        
        if not gex_by_strike:
            return result
        
        # 过滤并排序 - 只保留现价附近 ±10% 的 strikes
        valid_strikes = [s for s in gex_by_strike.keys() if spot * 0.9 <= s <= spot * 1.1]
        valid_strikes = sorted(valid_strikes)
        
        if not valid_strikes:
            return result
        
        for strike in valid_strikes:
            result['strikes'].append(strike)
            result['gex_call'].append(gex_by_strike[strike]['call'])
            result['gex_put'].append(-gex_by_strike[strike]['put'])  # Put GEX 为负
            result['gex_net'].append(gex_by_strike[strike]['call'] - gex_by_strike[strike]['put'])
        
        # 计算总 GEX
        result['total_gex'] = sum(result['gex_net'])
        
        # 找 Gamma Flip Point (净 GEX 从正变负或从负变正的点)
        for i in range(len(result['strikes']) - 1):
            current_gex = result['gex_net'][i]
            next_gex = result['gex_net'][i+1]
            if (current_gex > 0 and next_gex < 0) or (current_gex < 0 and next_gex > 0):
                result['gamma_flip'] = (result['strikes'][i] + result['strikes'][i+1]) / 2
                break
        
        # 找 Put Wall 和 Call Wall (最大 GEX 集中位置)
        if result['gex_call']:
            max_call_gex = max(result['gex_call'])
            if max_call_gex > 0:
                max_call_idx = result['gex_call'].index(max_call_gex)
                result['call_wall'] = result['strikes'][max_call_idx]
        
        if result['gex_put']:
            min_put_gex = min(result['gex_put'])  # 最负的
            if min_put_gex < 0:
                max_put_idx = result['gex_put'].index(min_put_gex)
                result['put_wall'] = result['strikes'][max_put_idx]
        
        # 计算 Max Pain (简化版 - 找 OI 最集中的 strike)
        try:
            total_oi_by_strike = {}
            for opt in all_options:
                strike = opt['strike']
                if strike in valid_strikes:
                    if strike not in total_oi_by_strike:
                        total_oi_by_strike[strike] = 0
                    total_oi_by_strike[strike] += opt['oi']
            
            if total_oi_by_strike:
                result['max_pain'] = max(total_oi_by_strike, key=total_oi_by_strike.get)
        except:
            pass
        
    except Exception as e:
        # 静默失败，返回空结果
        pass
    
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
st.title("🦅 宏观战情观察室")
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
    # 显示数据时间戳
    gex_time_col1, gex_time_col2, gex_time_col3 = st.columns(3)
    with gex_time_col1:
        st.caption(f"📅 OI 数据日期: **{gex_data.get('oi_date', 'N/A')}** ({gex_data.get('oi_weekday', '')})")
    with gex_time_col2:
        st.caption(f"⏰ 计算时间: **{gex_data.get('calc_time', 'N/A')}**")
    with gex_time_col3:
        st.caption("💡 OI 每天盘前更新，反映前一交易日收盘持仓")
    
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
# [导出功能已移至页面底部]

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



# ============================================================================
# 产品配置
# ============================================================================
PRODUCT_CONFIG = {
    'ES': {
        'name': 'ES (E-mini S&P 500)',
        'default_atr': 20,
        'price_format': '#.00',
        'description': 'ES价格约6000-7000点，ATR约15-25点'
    },
    'NQ': {
        'name': 'NQ (E-mini Nasdaq 100)',
        'default_atr': 80,
        'price_format': '#.00',
        'description': 'NQ价格约20000-22000点，ATR约60-100点'
    }
}

# ============================================================================
# 核心计算函数
# ============================================================================

def load_and_prepare_data(uploaded_file):
    """加载并准备数据"""
    df = pd.read_csv(uploaded_file)
    
    # 尝试多种日期格式
    try:
        df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d')
    except:
        try:
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d')
        except:
            df['time'] = pd.to_datetime(df['time'])
    
    df = df.sort_values('time').reset_index(drop=True)
    
    # 计算ATR(14)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # 计算ATR均值（用于判断ATR扩张）
    df['atr_ma'] = df['atr'].rolling(window=20).mean()
    
    return df


def find_swing_candidates(df, left_bars=3):
    """
    找出候选Swing点
    Swing High: 当日High > 前left_bars日所有High
    Swing Low: 当日Low < 前left_bars日所有Low
    """
    swing_highs = []
    swing_lows = []
    
    for i in range(left_bars, len(df)):
        # 检查Swing High
        current_high = df.iloc[i]['high']
        is_swing_high = True
        for j in range(1, left_bars + 1):
            if df.iloc[i - j]['high'] >= current_high:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append(i)
        
        # 检查Swing Low
        current_low = df.iloc[i]['low']
        is_swing_low = True
        for j in range(1, left_bars + 1):
            if df.iloc[i - j]['low'] <= current_low:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append(i)
    
    return swing_highs, swing_lows


def validate_directional_extension(df, idx, is_high, lookforward=5, atr_multiplier=1.5):
    """
    条件一：验证方向性延伸
    - 至少3-5根同方向K线
    - 总移动幅度 >= 1.5 × ATR
    - 未被快速完全反向吞没
    """
    if idx + lookforward >= len(df):
        return False, 0
    
    atr = df.iloc[idx]['atr']
    if pd.isna(atr):
        return False, 0
    
    required_move = atr * atr_multiplier
    
    if is_high:
        # Swing High后应该向下延伸
        start_price = df.iloc[idx]['high']
        min_price = df.iloc[idx + 1: idx + lookforward + 1]['low'].min()
        move = start_price - min_price
        
        # 检查是否被快速吞没（后续K线没有立即创新高）
        max_high_after = df.iloc[idx + 1: idx + lookforward + 1]['high'].max()
        if max_high_after > start_price:
            return False, 0
    else:
        # Swing Low后应该向上延伸
        start_price = df.iloc[idx]['low']
        max_price = df.iloc[idx + 1: idx + lookforward + 1]['high'].max()
        move = max_price - start_price
        
        # 检查是否被快速吞没
        min_low_after = df.iloc[idx + 1: idx + lookforward + 1]['low'].min()
        if min_low_after < start_price:
            return False, 0
    
    return move >= required_move, move


def validate_structure_break(df, swing_highs, swing_lows, idx, is_high):
    """
    条件二：打破前一轮结构
    Swing High有效：后续价格突破了前一个Lower High
    Swing Low有效：后续价格突破了前一个Higher Low
    """
    if is_high:
        prev_highs = [h for h in swing_highs if h < idx]
        if len(prev_highs) < 2:
            return True
        
        prev_high_idx = prev_highs[-1]
        prev_prev_high_idx = prev_highs[-2]
        
        current_high = df.iloc[idx]['high']
        prev_high = df.iloc[prev_high_idx]['high']
        prev_prev_high = df.iloc[prev_prev_high_idx]['high']
        
        if current_high > prev_high or (prev_high < prev_prev_high and current_high > prev_high):
            return True
    else:
        prev_lows = [l for l in swing_lows if l < idx]
        if len(prev_lows) < 2:
            return True
        
        prev_low_idx = prev_lows[-1]
        prev_prev_low_idx = prev_lows[-2]
        
        current_low = df.iloc[idx]['low']
        prev_low = df.iloc[prev_low_idx]['low']
        prev_prev_low = df.iloc[prev_prev_low_idx]['low']
        
        if current_low < prev_low or (prev_low > prev_prev_low and current_low < prev_low):
            return True
    
    return False


def check_volatility_expansion(df, idx):
    """
    条件三（加分项）：波动率扩张
    当日ATR > 1.3 × ATR均值
    """
    atr = df.iloc[idx]['atr']
    atr_ma = df.iloc[idx]['atr_ma']
    
    if pd.isna(atr) or pd.isna(atr_ma):
        return False
    
    return atr > atr_ma * 1.3


def classify_structure_level(df, idx, is_high, move_size, has_volatility_expansion):
    """
    结构分级
    一级：趋势起点/终点/反转点 + 波动率扩张
    二级：趋势中段回撤点
    """
    atr = df.iloc[idx]['atr']
    if pd.isna(atr):
        return 2
    
    if move_size > atr * 2 and has_volatility_expansion:
        return 1
    
    if move_size > atr * 2.5:
        return 1
    
    return 2


def calculate_zone(df, idx, is_high, zone_width_multiplier=0.3, default_atr=20):
    """
    计算Zone区间
    区间宽度 = 0.2-0.4 × ATR
    """
    atr = df.iloc[idx]['atr']
    if pd.isna(atr):
        atr = default_atr
    
    zone_width = atr * zone_width_multiplier
    
    if is_high:
        price = df.iloc[idx]['high']
        zone_top = price + zone_width / 2
        zone_bottom = price - zone_width / 2
    else:
        price = df.iloc[idx]['low']
        zone_top = price + zone_width / 2
        zone_bottom = price - zone_width / 2
    
    return zone_top, zone_bottom, price


def analyze_structures(df, default_atr=20, zone_width_multiplier=0.3):
    """
    主分析函数：识别所有合格结构位
    """
    swing_highs, swing_lows = find_swing_candidates(df, left_bars=3)
    
    structures = []
    
    # 分析Swing Highs
    for idx in swing_highs:
        valid_extension, move_size = validate_directional_extension(df, idx, is_high=True)
        if not valid_extension:
            continue
        
        valid_break = validate_structure_break(df, swing_highs, swing_lows, idx, is_high=True)
        if not valid_break:
            continue
        
        has_vol_expansion = check_volatility_expansion(df, idx)
        level = classify_structure_level(df, idx, is_high=True, move_size=move_size, has_volatility_expansion=has_vol_expansion)
        zone_top, zone_bottom, price = calculate_zone(df, idx, is_high=True, zone_width_multiplier=zone_width_multiplier, default_atr=default_atr)
        
        structures.append({
            'date': df.iloc[idx]['time'],
            'type': 'resistance',
            'level': level,
            'price': price,
            'zone_top': zone_top,
            'zone_bottom': zone_bottom,
            'move_size': move_size,
            'vol_expansion': has_vol_expansion,
            'idx': idx
        })
    
    # 分析Swing Lows
    for idx in swing_lows:
        valid_extension, move_size = validate_directional_extension(df, idx, is_high=False)
        if not valid_extension:
            continue
        
        valid_break = validate_structure_break(df, swing_highs, swing_lows, idx, is_high=False)
        if not valid_break:
            continue
        
        has_vol_expansion = check_volatility_expansion(df, idx)
        level = classify_structure_level(df, idx, is_high=False, move_size=move_size, has_volatility_expansion=has_vol_expansion)
        zone_top, zone_bottom, price = calculate_zone(df, idx, is_high=False, zone_width_multiplier=zone_width_multiplier, default_atr=default_atr)
        
        structures.append({
            'date': df.iloc[idx]['time'],
            'type': 'support',
            'level': level,
            'price': price,
            'zone_top': zone_top,
            'zone_bottom': zone_bottom,
            'move_size': move_size,
            'vol_expansion': has_vol_expansion,
            'idx': idx
        })
    
    return pd.DataFrame(structures)


def get_active_structures(structures_df, current_price):
    """
    获取当前仍然有效的结构位
    """
    if structures_df.empty:
        return pd.DataFrame()
    
    active = []
    
    for _, row in structures_df.iterrows():
        if row['type'] == 'resistance':
            if current_price < row['zone_top']:
                active.append(row)
        else:
            if current_price > row['zone_bottom']:
                active.append(row)
    
    return pd.DataFrame(active)


def format_output(structures_df, current_price, product):
    """
    格式化输出结果
    """
    if structures_df.empty:
        return "未找到有效结构位"
    
    resistances = structures_df[structures_df['type'] == 'resistance'].sort_values('price', ascending=True)
    supports = structures_df[structures_df['type'] == 'support'].sort_values('price', ascending=False)
    
    output_lines = []
    output_lines.append(f"{product} 结构位分析")
    output_lines.append(f"当前价格: {current_price:.2f}")
    output_lines.append("=" * 40)
    
    output_lines.append("\n📈 阻力位 (Resistance)")
    output_lines.append("-" * 40)
    
    r1_count = 0
    r2_count = 0
    for _, row in resistances.iterrows():
        level_str = "★一级" if row['level'] == 1 else "二级"
        vol_str = " [放量]" if row['vol_expansion'] else ""
        distance = row['price'] - current_price
        output_lines.append(
            f"{level_str}: {row['zone_bottom']:.2f} - {row['zone_top']:.2f} "
            f"(+{distance:.2f}点){vol_str}"
        )
        if row['level'] == 1:
            r1_count += 1
        else:
            r2_count += 1
    
    output_lines.append("\n📉 支撑位 (Support)")
    output_lines.append("-" * 40)
    
    s1_count = 0
    s2_count = 0
    for _, row in supports.iterrows():
        level_str = "★一级" if row['level'] == 1 else "二级"
        vol_str = " [放量]" if row['vol_expansion'] else ""
        distance = current_price - row['price']
        output_lines.append(
            f"{level_str}: {row['zone_bottom']:.2f} - {row['zone_top']:.2f} "
            f"(-{distance:.2f}点){vol_str}"
        )
        if row['level'] == 1:
            s1_count += 1
        else:
            s2_count += 1
    
    output_lines.append("\n" + "=" * 40)
    output_lines.append(f"统计: 一级阻力{r1_count}个, 二级阻力{r2_count}个, 一级支撑{s1_count}个, 二级支撑{s2_count}个")
    
    return "\n".join(output_lines)


def create_chart(df, structures_df, current_price, product):
    """
    创建K线图并标注结构位
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3])
    
    fig.add_trace(
        go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=product
        ),
        row=1, col=1
    )
    
    for _, row in structures_df.iterrows():
        color = 'rgba(255, 0, 0, 0.2)' if row['type'] == 'resistance' else 'rgba(0, 255, 0, 0.2)'
        border_color = 'red' if row['type'] == 'resistance' else 'green'
        line_width = 2 if row['level'] == 1 else 1
        
        fig.add_hrect(
            y0=row['zone_bottom'],
            y1=row['zone_top'],
            fillcolor=color,
            line=dict(color=border_color, width=line_width),
            row=1, col=1
        )
        
        level_str = "L1" if row['level'] == 1 else "L2"
        type_str = "R" if row['type'] == 'resistance' else "S"
        fig.add_annotation(
            x=df['time'].iloc[-1],
            y=row['price'],
            text=f"{type_str}{level_str}: {row['price']:.0f}",
            showarrow=False,
            xanchor='left',
            font=dict(size=10, color=border_color),
            row=1, col=1
        )
    
    fig.add_hline(y=current_price, line_dash="dash", line_color="blue",
                  annotation_text=f"当前: {current_price:.2f}", row=1, col=1)
    
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['atr'], name='ATR(14)', line=dict(color='orange')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['atr_ma'], name='ATR MA(20)', line=dict(color='gray', dash='dash')),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{product} 日线结构位分析',
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    return fig


# ============================================================================
# Streamlit 界面 - ES/NQ 结构位分析
# ============================================================================

st.divider()
st.header("9. 📊 ES/NQ 日线结构位分析器")
st.markdown("""
基于Swing High/Low识别有效结构位，输出Zone区间供日内交易参考。

**筛选条件：**
1. 方向性延伸 ≥ 1.5× ATR
2. 打破前一轮结构形态
3. 波动率扩张（加分项）
""")

# ES/NQ 分析器控制 (内置在主区域)
col_es1, col_es2, col_es3, col_es4, col_es5 = st.columns(5)
with col_es1:
    product = st.selectbox("选择产品", options=['ES', 'NQ'], format_func=lambda x: PRODUCT_CONFIG[x]['name'])
with col_es2:
    left_bars = st.slider("Swing左侧K线", 2, 5, 3)
with col_es3:
    lookforward = st.slider("延伸确认K线", 3, 7, 5)
with col_es4:
    atr_multiplier = st.slider("ATR倍数", 1.0, 2.5, 1.5)
with col_es5:
    zone_width = st.slider("Zone宽度", 0.2, 0.5, 0.3)

# 获取产品配置
config = PRODUCT_CONFIG[product]

# 文件上传
st.subheader(f"📁 上传 {product} 日线数据")
uploaded_file = st.file_uploader(f"上传{product}日线CSV文件", type=['csv'])

if uploaded_file is not None:
    # 加载数据
    df = load_and_prepare_data(uploaded_file)
    
    st.success(f"✅ 数据加载成功: {len(df)}个交易日 ({df['time'].min().strftime('%Y-%m-%d')} 至 {df['time'].max().strftime('%Y-%m-%d')})")
    
    # 当前价格和ATR
    current_price = df.iloc[-1]['close']
    current_atr = df.iloc[-1]['atr']
    
    st.info(f"📊 **{product}** | 当前价格: {current_price:.2f} | ATR(14): {current_atr:.2f} 点 | Zone宽度约: {current_atr * zone_width:.2f} 点")
    
    # 分析结构
    with st.spinner("正在分析结构位..."):
        all_structures = analyze_structures(df, default_atr=config['default_atr'], zone_width_multiplier=zone_width)
        active_structures = get_active_structures(all_structures, current_price)
    
    # 显示结果
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 当前有效结构位")
        output_text = format_output(active_structures, current_price, product)
        st.code(output_text, language=None)
        
        # TradingView输入格式
        st.subheader("📝 TradingView快速输入")
        if not active_structures.empty:
            tv_lines = []
            resistances = active_structures[active_structures['type'] == 'resistance'].sort_values('price')
            supports = active_structures[active_structures['type'] == 'support'].sort_values('price', ascending=False)
            
            for i, (_, row) in enumerate(resistances.head(2).iterrows()):
                tv_lines.append(f"R{i+1}_top = {row['zone_top']:.2f}")
                tv_lines.append(f"R{i+1}_bottom = {row['zone_bottom']:.2f}")
            
            for i, (_, row) in enumerate(supports.head(2).iterrows()):
                tv_lines.append(f"S{i+1}_top = {row['zone_top']:.2f}")
                tv_lines.append(f"S{i+1}_bottom = {row['zone_bottom']:.2f}")
            
            st.code("\n".join(tv_lines), language=None)
    
    with col2:
        st.subheader("📈 K线图")
        fig = create_chart(df, active_structures, current_price, product)
        st.plotly_chart(fig, use_container_width=True)
    
    # 详细数据表
    with st.expander("查看所有检测到的结构位"):
        if not all_structures.empty:
            display_df = all_structures.copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df['level'] = display_df['level'].map({1: '一级', 2: '二级'})
            display_df['type'] = display_df['type'].map({'resistance': '阻力', 'support': '支撑'})
            display_df = display_df[['date', 'type', 'level', 'price', 'zone_top', 'zone_bottom', 'move_size', 'vol_expansion']]
            display_df.columns = ['日期', '类型', '级别', '价格', 'Zone上沿', 'Zone下沿', '延伸幅度', '放量']
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("未检测到符合条件的结构位")

else:
    st.info(f"👆 请先在左侧选择产品（ES/NQ），然后上传对应的日线CSV文件")
    
    st.markdown("""
    ### 如何导出数据
    1. 在TradingView打开对应产品的日线图
       - ES: `ES1!` 或 `ESH2025`
       - NQ: `NQ1!` 或 `NQH2025`
    2. 确保时间框架选择 **1D (日线)**
    3. 图表右上角菜单 → **Export chart data**
    4. 下载CSV文件并上传到这里
    
    ### CSV格式要求
    ```
    time,open,high,low,close,Volume
    2025/6/2,5898.75,5955.5,5867.5,5947.25,1194125
    ...
    ```
    
    ### ES vs NQ 参考
    | 产品 | 价格范围 | ATR范围 | Zone宽度 |
    |------|----------|---------|----------|
    | ES | 6000-7000 | 15-25点 | ~6点 |
    | NQ | 20000-22000 | 60-100点 | ~24点 |
    """)
    

# ============================================================================
# 📊 资金轮动评分系统 (Rotation Score System)
# 添加到现有 app.py 末尾
# ============================================================================

st.divider()
st.header("10. 📊 资金轮动趋势评分 (Rotation Score)")

st.markdown("""
<div class="summary-box summary-neutral">
<b>📈 趋势评分系统</b>：基于多因子模型计算市场资金流向，输出 -100 到 +100 的综合评分。
正值 = Risk-On (进攻)，负值 = Risk-Off (防御)。结合 Gamma 环境使用效果最佳。
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 配置区：因子定义
# ============================================================================

ROTATION_FACTORS = {
    'risk_appetite': {
        'name': '风险偏好',
        'weight': 0.35,
        'pairs': [
            {'name': 'Beta_Trade', 'numerator': 'SPHB', 'denominator': 'SPLV', 'weight': 0.3, 'desc': '高贝塔/低波动'},
            {'name': 'Growth_Value', 'numerator': 'IWF', 'denominator': 'IWD', 'weight': 0.25, 'desc': '成长/价值'},
            {'name': 'Credit_Spread', 'numerator': 'HYG', 'denominator': 'IEF', 'weight': 0.25, 'desc': '垃圾债/国债'},
            {'name': 'Speculative', 'numerator': 'ARKK', 'denominator': 'QQQ', 'weight': 0.2, 'desc': '投机/主流'},
        ]
    },
    'sector_rotation': {
        'name': '板块轮动',
        'weight': 0.40,
        'pairs': [
            {'name': 'Tech_Staples', 'numerator': 'XLK', 'denominator': 'XLP', 'weight': 0.25, 'desc': '科技/必需'},
            {'name': 'Semis_Alpha', 'numerator': 'SMH', 'denominator': 'QQQ', 'weight': 0.25, 'desc': '半导体/纳指'},
            {'name': 'Software_Alpha', 'numerator': 'IGV', 'denominator': 'QQQ', 'weight': 0.20, 'desc': '软件/纳指'},
            {'name': 'Cyclical_Defensive', 'numerator': 'XLY', 'denominator': 'XLU', 'weight': 0.15, 'desc': '可选/公用'},
            {'name': 'Financials', 'numerator': 'XLF', 'denominator': 'SPY', 'weight': 0.15, 'desc': '金融/大盘'},
        ]
    },
    'liquidity': {
        'name': '流动性广度',
        'weight': 0.25,
        'pairs': [
            {'name': 'Small_Large', 'numerator': 'IWM', 'denominator': 'SPY', 'weight': 0.35, 'desc': '小盘/大盘'},
            {'name': 'Equal_Cap', 'numerator': 'RSP', 'denominator': 'SPY', 'weight': 0.35, 'desc': '等权/市值'},
            {'name': 'EM_US', 'numerator': 'EEM', 'denominator': 'SPY', 'weight': 0.30, 'desc': '新兴/美股'},
        ]
    }
}

LOOKBACK_PERIOD = 20  # Z-Score 计算窗口
Z_SCORE_CLIP = 3.0    # 极值处理

# ============================================================================
# 数据获取函数
# ============================================================================

@st.cache_data(ttl=3600)  # 缓存1小时
def get_rotation_data(tickers: list, period: str = "60d") -> pd.DataFrame:
    """获取所有需要的 ETF 数据"""
    try:
        data = yf.download(tickers, period=period, progress=False)
        
        # 检查是否为空
        if data.empty:
            st.warning("Yahoo Finance 返回空数据")
            return pd.DataFrame()
        
        # 处理 MultiIndex 格式 (新版 yfinance)
        # Level 0 = 价格类型 (Close, Adj Close, etc.)
        # Level 1 = Ticker 名称
        if isinstance(data.columns, pd.MultiIndex):
            # 优先使用 Adj Close，否则用 Close
            if 'Adj Close' in data.columns.get_level_values(0):
                return data['Adj Close']
            elif 'Close' in data.columns.get_level_values(0):
                return data['Close']
            else:
                st.warning("数据中没有 Close 或 Adj Close 列")
                return pd.DataFrame()
        else:
            # 单 ticker 或旧版格式
            if 'Adj Close' in data.columns:
                return data[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
            elif 'Close' in data.columns:
                return data[['Close']].rename(columns={'Close': tickers[0]})
            return data
            
    except Exception as e:
        st.warning(f"数据获取失败: {e}")
        return pd.DataFrame()

def calculate_z_score(series: pd.Series, lookback: int = 20) -> float:
    """计算 Z-Score"""
    if len(series) < lookback:
        return 0.0
    
    recent = series.iloc[-lookback:]
    current = series.iloc[-1]
    mean = recent.mean()
    std = recent.std()
    
    if std == 0:
        return 0.0
    
    z = (current - mean) / std
    return np.clip(z, -Z_SCORE_CLIP, Z_SCORE_CLIP)

def calculate_ratio_z_score(data: pd.DataFrame, numerator: str, denominator: str, lookback: int = 20) -> tuple:
    """计算比率的 Z-Score"""
    if numerator not in data.columns or denominator not in data.columns:
        return 0.0, 0.0, 0.0
    
    ratio = data[numerator] / data[denominator]
    ratio = ratio.dropna()
    
    if len(ratio) < lookback:
        return 0.0, 0.0, 0.0
    
    current_ratio = ratio.iloc[-1]
    z_score = calculate_z_score(ratio, lookback)
    
    # 计算5日变化
    if len(ratio) >= 5:
        change_5d = (ratio.iloc[-1] / ratio.iloc[-5] - 1) * 100
    else:
        change_5d = 0.0
    
    return z_score, current_ratio, change_5d

def calculate_rotation_score(data: pd.DataFrame) -> dict:
    """计算综合 Rotation Score"""
    results = {
        'total_score': 0.0,
        'categories': {},
        'factors': [],
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    
    total_score = 0.0
    
    for cat_key, category in ROTATION_FACTORS.items():
        cat_score = 0.0
        cat_factors = []
        
        for pair in category['pairs']:
            z_score, ratio, change_5d = calculate_ratio_z_score(
                data, pair['numerator'], pair['denominator'], LOOKBACK_PERIOD
            )
            
            weighted_z = z_score * pair['weight']
            cat_score += weighted_z
            
            factor_result = {
                'name': pair['name'],
                'desc': pair['desc'],
                'pair': f"{pair['numerator']}/{pair['denominator']}",
                'ratio': ratio,
                'z_score': z_score,
                'change_5d': change_5d,
                'weighted': weighted_z,
                'signal': 'bullish' if z_score > 0.5 else 'bearish' if z_score < -0.5 else 'neutral'
            }
            cat_factors.append(factor_result)
            results['factors'].append(factor_result)
        
        # 归一化到 -100 ~ +100
        cat_normalized = (cat_score / Z_SCORE_CLIP) * 100
        results['categories'][cat_key] = {
            'name': category['name'],
            'score': cat_normalized,
            'weight': category['weight'],
            'factors': cat_factors
        }
        
        total_score += cat_normalized * category['weight']
    
    results['total_score'] = np.clip(total_score, -100, 100)
    
    return results

# ============================================================================
# Gamma 数据解析
# ============================================================================

def parse_gamma_input(text: str) -> dict:
    """解析 SpotGamma 粘贴数据"""
    result = {
        'zero_gamma': None,
        'call_wall': None,
        'put_wall': None,
        'vol_trigger': None,
        'dex': None,
        'gex': None,
        'levels': []
    }
    
    if not text or not text.strip():
        return result
    
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 尝试解析 DEX/GEX
        line_lower = line.lower()
        if 'dex' in line_lower:
            try:
                # 提取数字
                import re
                numbers = re.findall(r'-?\d+\.?\d*', line)
                if numbers:
                    result['dex'] = float(numbers[0])
            except:
                pass
            continue
        
        if 'gex' in line_lower and 'zero' not in line_lower:
            try:
                import re
                numbers = re.findall(r'-?\d+\.?\d*', line)
                if numbers:
                    result['gex'] = float(numbers[0])
            except:
                pass
            continue
        
        # 解析价位数据 (Tab 或空格分隔)
        parts = line.replace('\t', ' ').split()
        if len(parts) >= 2:
            try:
                price = float(parts[0].replace(',', ''))
                level_name = ' '.join(parts[1:]).lower()
                
                result['levels'].append({'price': price, 'name': ' '.join(parts[1:])})
                
                if 'zero gamma' in level_name:
                    result['zero_gamma'] = price
                elif 'call wall' in level_name:
                    result['call_wall'] = price
                elif 'put wall' in level_name:
                    result['put_wall'] = price
                elif 'vol' in level_name and 'trigger' in level_name:
                    result['vol_trigger'] = price
            except ValueError:
                continue
    
    return result

# ============================================================================
# 策略建议生成
# ============================================================================

def generate_strategy_recommendation(rotation_score: float, gamma_data: dict, current_price: float = None) -> dict:
    """基于 Rotation Score 和 Gamma 环境生成策略建议"""
    
    rec = {
        'market_state': '',
        'gamma_env': '',
        'strategy': '',
        'reasoning': [],
        'risk_level': ''
    }
    
    # 判断 Rotation 状态
    if rotation_score > 60:
        rec['market_state'] = '强力进攻 (Strong Risk-On)'
        rot_bias = 'very_bullish'
    elif rotation_score > 20:
        rec['market_state'] = '震荡偏多 (Mild Risk-On)'
        rot_bias = 'bullish'
    elif rotation_score > -20:
        rec['market_state'] = '无序震荡 (Neutral)'
        rot_bias = 'neutral'
    elif rotation_score > -60:
        rec['market_state'] = '避险调整 (Mild Risk-Off)'
        rot_bias = 'bearish'
    else:
        rec['market_state'] = '恐慌抛售 (Strong Risk-Off)'
        rot_bias = 'very_bearish'
    
    # 判断 Gamma 环境
    gamma_env = 'unknown'
    if gamma_data.get('zero_gamma') and current_price:
        if current_price > gamma_data['zero_gamma']:
            gamma_env = 'positive'
            rec['gamma_env'] = '正 Gamma (做市商高抛低吸)'
        else:
            gamma_env = 'negative'
            rec['gamma_env'] = '负 Gamma (做市商追涨杀跌)'
    
    # 生成策略建议
    strategy_matrix = {
        ('very_bullish', 'positive'): {
            'strategy': 'Bull Call Spread',
            'reasoning': ['资金强势流入进攻板块', '正 Gamma 压制暴涨，Spread 合适', '目标 Call Wall'],
            'risk': '中等'
        },
        ('very_bullish', 'negative'): {
            'strategy': 'Long Call',
            'reasoning': ['资金强势流入', '负 Gamma 可能暴涨', '不限制上涨收益'],
            'risk': '较高'
        },
        ('bullish', 'positive'): {
            'strategy': 'Bull Call Spread',
            'reasoning': ['资金偏多', '正 Gamma 震荡上行', '控制成本'],
            'risk': '中低'
        },
        ('bullish', 'negative'): {
            'strategy': 'Bull Call Spread / Long Call',
            'reasoning': ['资金偏多', '负 Gamma 波动大', '视风险偏好选择'],
            'risk': '中高'
        },
        ('neutral', 'positive'): {
            'strategy': 'Iron Condor / 观望',
            'reasoning': ['资金方向不明', '正 Gamma 震荡', '卖两边收权利金'],
            'risk': '中等'
        },
        ('neutral', 'negative'): {
            'strategy': 'Long Straddle / 观望',
            'reasoning': ['方向不明但波动大', '负 Gamma 等突破', '买两边'],
            'risk': '较高'
        },
        ('bearish', 'positive'): {
            'strategy': 'Bear Put Spread',
            'reasoning': ['资金流出', '正 Gamma 慢跌', 'Spread 控制成本'],
            'risk': '中等'
        },
        ('bearish', 'negative'): {
            'strategy': 'Long Put',
            'reasoning': ['资金流出', '负 Gamma 可能暴跌', '不限制下跌收益'],
            'risk': '较高'
        },
        ('very_bearish', 'positive'): {
            'strategy': 'Bear Put Spread',
            'reasoning': ['资金恐慌流出', '但正 Gamma 有支撑', '控制风险'],
            'risk': '中高'
        },
        ('very_bearish', 'negative'): {
            'strategy': 'Long Put / Long Straddle',
            'reasoning': ['极度恐慌', '负 Gamma 可能崩盘', '做空或做波动率'],
            'risk': '高'
        },
    }
    
    key = (rot_bias, gamma_env)
    if key in strategy_matrix:
        rec['strategy'] = strategy_matrix[key]['strategy']
        rec['reasoning'] = strategy_matrix[key]['reasoning']
        rec['risk_level'] = strategy_matrix[key]['risk']
    else:
        rec['strategy'] = '观望 / 轻仓试探'
        rec['reasoning'] = ['Gamma 数据不完整', '建议等待更多信息']
        rec['risk_level'] = '未知'
    
    return rec

# ============================================================================
# 导出函数
# ============================================================================

def generate_rotation_export(results: dict, gamma_qqq: dict, gamma_nq: dict, gamma_ndx: dict, 
                            recommendation: dict, current_prices: dict) -> str:
    """生成 Claude 导出文本"""
    
    export_lines = [
        "# 📊 资金轮动分析报告",
        f"生成时间: {results['timestamp']}",
        "",
        "## 一、综合评分",
        f"**Rotation Score: {results['total_score']:.1f}** (-100 到 +100)",
        "",
    ]
    
    # 分类评分
    export_lines.append("## 二、分类评分")
    for cat_key, cat_data in results['categories'].items():
        export_lines.append(f"### {cat_data['name']} (权重 {cat_data['weight']*100:.0f}%)")
        export_lines.append(f"评分: {cat_data['score']:.1f}")
        export_lines.append("")
        for factor in cat_data['factors']:
            signal_emoji = '🟢' if factor['signal'] == 'bullish' else '🔴' if factor['signal'] == 'bearish' else '⚪'
            export_lines.append(f"- {signal_emoji} {factor['desc']} ({factor['pair']}): Z={factor['z_score']:.2f}, 5D变化={factor['change_5d']:.2f}%")
        export_lines.append("")
    
    # Gamma 数据
    export_lines.append("## 三、Gamma 环境")
    
    if gamma_qqq.get('zero_gamma'):
        export_lines.append("### QQQ")
        export_lines.append(f"- 当前价: ${current_prices.get('QQQ', 'N/A')}")
        export_lines.append(f"- Zero Gamma: ${gamma_qqq.get('zero_gamma')}")
        export_lines.append(f"- Call Wall: ${gamma_qqq.get('call_wall')}")
        export_lines.append(f"- Put Wall: ${gamma_qqq.get('put_wall')}")
        if gamma_qqq.get('dex'):
            export_lines.append(f"- DEX: {gamma_qqq.get('dex')}M")
        if gamma_qqq.get('gex'):
            export_lines.append(f"- GEX: {gamma_qqq.get('gex')}M")
        export_lines.append("")
    
    if gamma_nq.get('zero_gamma'):
        export_lines.append("### NQ")
        export_lines.append(f"- 当前价: {current_prices.get('NQ', 'N/A')}")
        export_lines.append(f"- Zero Gamma: {gamma_nq.get('zero_gamma')}")
        export_lines.append(f"- Call Wall: {gamma_nq.get('call_wall')}")
        export_lines.append(f"- Put Wall: {gamma_nq.get('put_wall')}")
        export_lines.append("")
    
    if gamma_ndx.get('zero_gamma'):
        export_lines.append("### NDX")
        export_lines.append(f"- Zero Gamma: {gamma_ndx.get('zero_gamma')}")
        export_lines.append(f"- Call Wall: {gamma_ndx.get('call_wall')}")
        export_lines.append(f"- Put Wall: {gamma_ndx.get('put_wall')}")
        if gamma_ndx.get('dex'):
            export_lines.append(f"- DEX: {gamma_ndx.get('dex')}M")
        if gamma_ndx.get('gex'):
            export_lines.append(f"- GEX: {gamma_ndx.get('gex')}M")
        export_lines.append("")
    
    # 策略建议
    export_lines.append("## 四、策略建议")
    export_lines.append(f"- 市场状态: {recommendation['market_state']}")
    export_lines.append(f"- Gamma 环境: {recommendation['gamma_env']}")
    export_lines.append(f"- 推荐策略: **{recommendation['strategy']}**")
    export_lines.append(f"- 风险等级: {recommendation['risk_level']}")
    export_lines.append("- 理由:")
    for reason in recommendation['reasoning']:
        export_lines.append(f"  - {reason}")
    
    export_lines.append("")
    export_lines.append("---")
    export_lines.append("请基于以上数据进行深度分析:")
    export_lines.append("1. 资金流向是否支持当前趋势？")
    export_lines.append("2. 哪些因子是主要驱动力？")
    export_lines.append("3. Gamma 环境与资金流向是否共振？")
    export_lines.append("4. 建议的期权策略行权价？")
    
    return '\n'.join(export_lines)

# ============================================================================
# UI 部分
# ============================================================================

# Rotation Score 输入区域 (主区域内)
with st.expander("⚙️ Rotation Score 设置 & SpotGamma 数据输入", expanded=True):
    col_price1, col_price2, col_price3 = st.columns([1, 1, 2])
    
    with col_price1:
        st.markdown("**当前价格**")
        input_qqq_price = st.number_input("QQQ 价格", value=622.0, step=0.5, format="%.2f", key="rot_qqq_price")
        input_nq_price = st.number_input("NQ 价格", value=25800.0, step=10.0, format="%.2f", key="rot_nq_price")
    
    with col_price2:
        st.markdown("**QQQ Gamma**")
        gamma_qqq_input = st.text_area(
            "粘贴 QQQ SpotGamma",
            height=180,
            placeholder="621  Zero Gamma\n625  Call Wall\n590  Put Wall\nDEX: -1219.8\nGEX: 513",
            key="gamma_qqq_rot"
        )
    
    with col_price3:
        st.markdown("**NQ/NDX Gamma (三列格式)**")
        gamma_nq_ndx_input = st.text_area(
            "粘贴 NQ/NDX SpotGamma",
            height=180,
            placeholder="NDX      /NQ      Level ID\n25480    25718    Large Gamma 4\n25470    25708    Call Wall\n25250    25488    Large Gamma 1\n25170    25408    Volatility Trigger\n25150    25388    Put Wall\n25092    25330    Zero Gamma",
            key="gamma_nq_ndx_rot"
        )
        st.caption("格式: NDX [Tab/空格] NQ [Tab/空格] Level ID")

# 解析 Gamma 数据
gamma_qqq = parse_gamma_input(gamma_qqq_input)

# 解析 NQ/NDX 合并格式
def parse_nq_ndx_combined(text: str) -> tuple:
    """解析 NQ/NDX 三列格式数据"""
    result_nq = {
        'zero_gamma': None, 'call_wall': None, 'put_wall': None,
        'vol_trigger': None, 'levels': []
    }
    result_ndx = {
        'zero_gamma': None, 'call_wall': None, 'put_wall': None,
        'vol_trigger': None, 'levels': []
    }
    
    if not text or not text.strip():
        return result_nq, result_ndx
    
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or 'NDX' in line.upper() and '/NQ' in line.upper():  # 跳过标题行
            continue
        
        # 分割行 (Tab 或多空格)
        import re
        parts = re.split(r'\t+|\s{2,}', line)
        
        if len(parts) >= 3:
            try:
                ndx_price = float(parts[0].replace(',', ''))
                nq_price = float(parts[1].replace(',', '').replace('/',''))
                level_name = ' '.join(parts[2:]).lower()
                
                # 存储到对应的结果
                result_ndx['levels'].append({'price': ndx_price, 'name': parts[2]})
                result_nq['levels'].append({'price': nq_price, 'name': parts[2]})
                
                # 识别关键位置
                if 'zero gamma' in level_name:
                    result_ndx['zero_gamma'] = ndx_price
                    result_nq['zero_gamma'] = nq_price
                elif 'call wall' in level_name:
                    result_ndx['call_wall'] = ndx_price
                    result_nq['call_wall'] = nq_price
                elif 'put wall' in level_name:
                    result_ndx['put_wall'] = ndx_price
                    result_nq['put_wall'] = nq_price
                elif 'vol' in level_name and 'trigger' in level_name:
                    result_ndx['vol_trigger'] = ndx_price
                    result_nq['vol_trigger'] = nq_price
            except ValueError:
                continue
    
    return result_nq, result_ndx

gamma_nq, gamma_ndx = parse_nq_ndx_combined(gamma_nq_ndx_input)

# 存储到 session_state 供导出使用
st.session_state['gamma_qqq_data'] = gamma_qqq
st.session_state['gamma_nq_data'] = gamma_nq
st.session_state['gamma_ndx_data'] = gamma_ndx

current_prices = {
    'QQQ': input_qqq_price,
    'NQ': input_nq_price
}

# 收集所有需要的 ticker
all_tickers = set()
for category in ROTATION_FACTORS.values():
    for pair in category['pairs']:
        all_tickers.add(pair['numerator'])
        all_tickers.add(pair['denominator'])

all_tickers = list(all_tickers)

# 获取数据并计算
with st.spinner("正在获取 ETF 数据..."):
    rotation_data = get_rotation_data(all_tickers)

if not rotation_data.empty:
    results = calculate_rotation_score(rotation_data)
    st.session_state['rotation_results'] = results  # 存储到 session_state 供导出使用
    
    # 生成策略建议
    recommendation = generate_strategy_recommendation(
        results['total_score'], 
        gamma_qqq, 
        input_qqq_price
    )
    
    # ========================================
    # 主仪表盘
    # ========================================
    
    st.subheader("📈 综合评分仪表盘")
    
    col_gauge, col_status = st.columns([2, 1])
    
    with col_gauge:
        # 创建仪表盘
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=results['total_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Rotation Score", 'font': {'size': 24}},
            delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [-100, -60], 'color': '#dc3545'},
                    {'range': [-60, -20], 'color': '#fd7e14'},
                    {'range': [-20, 20], 'color': '#ffc107'},
                    {'range': [20, 60], 'color': '#90EE90'},
                    {'range': [60, 100], 'color': '#28a745'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': results['total_score']
                }
            }
        ))
        
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_status:
        # 状态卡片
        score = results['total_score']
        if score > 60:
            status_class = "summary-bull"
            status_emoji = "🚀"
            status_text = "强力进攻"
        elif score > 20:
            status_class = "summary-bull"
            status_emoji = "📈"
            status_text = "震荡偏多"
        elif score > -20:
            status_class = "summary-neutral"
            status_emoji = "⚖️"
            status_text = "无序震荡"
        elif score > -60:
            status_class = "summary-bear"
            status_emoji = "📉"
            status_text = "避险调整"
        else:
            status_class = "summary-bear"
            status_emoji = "🔻"
            status_text = "恐慌抛售"
        
        st.markdown(f"""
        <div class="summary-box {status_class}">
        <h2>{status_emoji} {status_text}</h2>
        <p><b>评分：{score:.1f}</b></p>
        <p>更新时间：{results['timestamp']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 分类评分
        st.markdown("**分类评分：**")
        for cat_key, cat_data in results['categories'].items():
            cat_score = cat_data['score']
            color = "green" if cat_score > 10 else "red" if cat_score < -10 else "gray"
            st.markdown(f"- {cat_data['name']}: <span style='color:{color}'><b>{cat_score:.1f}</b></span>", unsafe_allow_html=True)
    
    # ========================================
    # 因子详情
    # ========================================
    
    st.subheader("📊 因子分解")
    
    # 水平条形图
    factor_names = []
    factor_zscores = []
    factor_colors = []
    
    for factor in results['factors']:
        factor_names.append(f"{factor['desc']}")
        factor_zscores.append(factor['z_score'])
        factor_colors.append('#28a745' if factor['z_score'] > 0 else '#dc3545')
    
    fig_factors = go.Figure()
    fig_factors.add_trace(go.Bar(
        y=factor_names,
        x=factor_zscores,
        orientation='h',
        marker_color=factor_colors,
        text=[f"{z:.2f}" for z in factor_zscores],
        textposition='outside'
    ))
    
    fig_factors.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_factors.add_vline(x=0.5, line_dash="dot", line_color="green", opacity=0.5)
    fig_factors.add_vline(x=-0.5, line_dash="dot", line_color="red", opacity=0.5)
    
    fig_factors.update_layout(
        title="因子 Z-Score (偏离均值程度)",
        xaxis_title="Z-Score",
        yaxis_title="",
        height=400,
        xaxis=dict(range=[-3.5, 3.5]),
        showlegend=False
    )
    
    st.plotly_chart(fig_factors, use_container_width=True)
    
    # 因子详情表格
    with st.expander("📋 因子详情表"):
        factor_df = pd.DataFrame(results['factors'])
        factor_df = factor_df[['desc', 'pair', 'ratio', 'z_score', 'change_5d', 'signal']]
        factor_df.columns = ['因子', '比率对', '当前值', 'Z-Score', '5日变化%', '信号']
        factor_df['当前值'] = factor_df['当前值'].apply(lambda x: f"{x:.4f}" if x else "N/A")
        factor_df['Z-Score'] = factor_df['Z-Score'].apply(lambda x: f"{x:.2f}")
        factor_df['5日变化%'] = factor_df['5日变化%'].apply(lambda x: f"{x:.2f}%")
        factor_df['信号'] = factor_df['信号'].map({'bullish': '🟢 看涨', 'bearish': '🔴 看跌', 'neutral': '⚪ 中性'})
        st.dataframe(factor_df, use_container_width=True, hide_index=True)
    
    # ========================================
    # Gamma 环境与策略建议
    # ========================================
    
    st.subheader("🎯 Gamma 环境与策略建议")
    
    col_gamma, col_strategy = st.columns([1, 1])
    
    with col_gamma:
        st.markdown("**Gamma 关键位：**")
        
        # QQQ
        if gamma_qqq.get('zero_gamma'):
            qqq_pos = "✅ 正 Gamma" if input_qqq_price > gamma_qqq['zero_gamma'] else "⚠️ 负 Gamma"
            st.markdown(f"""
            **QQQ** ${input_qqq_price:.2f} | {qqq_pos}
            - Zero Gamma: ${gamma_qqq.get('zero_gamma')}
            - Call Wall: ${gamma_qqq.get('call_wall')}
            - Put Wall: ${gamma_qqq.get('put_wall')}
            """)
            if gamma_qqq.get('dex'):
                dex_sign = "📈" if gamma_qqq['dex'] > 0 else "📉"
                st.markdown(f"- DEX: {dex_sign} {gamma_qqq['dex']}M")
            if gamma_qqq.get('gex'):
                gex_sign = "🟢" if gamma_qqq['gex'] > 0 else "🔴"
                st.markdown(f"- GEX: {gex_sign} {gamma_qqq['gex']}M")
        else:
            st.info("👈 请在侧边栏粘贴 QQQ Gamma 数据")
        
        st.markdown("---")
        
        # NQ
        if gamma_nq.get('zero_gamma'):
            nq_pos = "✅ 正 Gamma" if input_nq_price > gamma_nq['zero_gamma'] else "⚠️ 负 Gamma"
            st.markdown(f"""
            **NQ** {input_nq_price:.2f} | {nq_pos}
            - Zero Gamma: {gamma_nq.get('zero_gamma')}
            - Call Wall: {gamma_nq.get('call_wall')}
            - Put Wall: {gamma_nq.get('put_wall')}
            """)
        else:
            st.info("👈 请在侧边栏粘贴 NQ Gamma 数据")
    
    with col_strategy:
        st.markdown("**📋 策略建议：**")
        
        if recommendation['strategy']:
            risk_color = {
                '中低': 'green',
                '中等': 'orange', 
                '中高': 'darkorange',
                '较高': 'red',
                '高': 'darkred'
            }.get(recommendation['risk_level'], 'gray')
            
            st.markdown(f"""
            <div class="summary-box summary-neutral">
            <h3>💡 {recommendation['strategy']}</h3>
            <p><b>市场状态：</b>{recommendation['market_state']}</p>
            <p><b>Gamma 环境：</b>{recommendation['gamma_env'] or '数据不足'}</p>
            <p><b>风险等级：</b><span style='color:{risk_color}'>{recommendation['risk_level']}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**理由：**")
            for reason in recommendation['reasoning']:
                st.markdown(f"- {reason}")
        else:
            st.info("请输入 Gamma 数据以获取策略建议")
    
    # [已移至页面底部统一导出]

else:
    st.warning("⚠️ 无法获取 ETF 数据，请检查网络连接或稍后重试")
    st.info("💡 如果问题持续，可能是 Yahoo Finance API 暂时不可用")
    results = None

# ============================================================================
# 市场广度雷达图 (可选)
# ============================================================================

if results is not None and 'categories' in results:
    with st.expander("📡 市场广度雷达图", expanded=False):
        # 提取各分类的分数
        categories = list(results['categories'].keys())
        cat_scores = [results['categories'][c]['score'] for c in categories]
        cat_names = [results['categories'][c]['name'] for c in categories]
        
        # 闭合雷达图
        cat_names_closed = cat_names + [cat_names[0]]
        cat_scores_closed = cat_scores + [cat_scores[0]]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=cat_scores_closed,
            theta=cat_names_closed,
            fill='toself',
            name='当前',
            line_color='blue'
        ))
        
        # 添加零线
        fig_radar.add_trace(go.Scatterpolar(
            r=[0] * len(cat_names_closed),
            theta=cat_names_closed,
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='中性线'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-100, 100]
                )
            ),
            showlegend=True,
            title="资金流向雷达图"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)

st.divider()
st.caption("📊 Rotation Score 系统 v1.0 | 数据来源: Yahoo Finance | 仅供参考，不构成投资建议")


# ============================================================================
# 11. 📊 ETF 板块资金流入扫描
# ============================================================================

st.divider()
st.header("11. 📊 ETF 板块资金流入扫描")

# 核心板块ETF列表
SECTOR_ETFS = {
    'XLK': '科技', 'SMH': '半导体', 'XLF': '金融', 'XLE': '能源',
    'XLV': '医疗健康', 'XBI': '生物科技', 'XLI': '工业', 'XLY': '可选消费',
    'XLP': '必需消费', 'XLU': '公用事业', 'XLRE': '房地产', 'XLB': '材料',
    'XLC': '通信服务', 'IWM': '小盘股', 'QQQ': '纳指100', 'SPY': 'S&P500',
}

@st.cache_data(ttl=300)
def get_etf_flow_data(ticker: str, period: str = "3mo"):
    """获取ETF数据用于资金流分析"""
    try:
        data = yf.download(ticker, period=period, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None

def calculate_etf_signals(ticker: str, data: pd.DataFrame) -> dict:
    """计算单个ETF的资金流入信号"""
    try:
        if data is None or data.empty or len(data) < 25:
            return None
        
        df = data.copy()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        latest = df.iloc[-1]
        prev_5d = df.iloc[-5]
        prev_20d = df.iloc[-20] if len(df) >= 20 else df.iloc[0]
        
        close = float(latest['Close'])
        sma20 = float(latest['SMA20'])
        sma50 = float(latest['SMA50']) if not pd.isna(latest['SMA50']) else sma20
        vol_sma20 = float(latest['Vol_SMA20'])
        
        price_above_sma20 = close > sma20
        price_above_sma50 = close > sma50
        volume_expanding = float(latest['Volume']) > vol_sma20
        obv_rising = float(latest['OBV']) > float(prev_5d['OBV'])
        returns_20d = (close / float(prev_20d['Close']) - 1) * 100
        vol_ratio = float(latest['Volume']) / vol_sma20 if vol_sma20 > 0 else 1
        
        score = sum([price_above_sma20, price_above_sma50, volume_expanding, obv_rising, returns_20d > 0])
        
        return {
            'ETF': ticker, '板块': SECTOR_ETFS.get(ticker, ticker),
            '价格': round(close, 2),
            '>SMA20': '✅' if price_above_sma20 else '❌',
            '>SMA50': '✅' if price_above_sma50 else '❌',
            '放量': '✅' if volume_expanding else '❌',
            'OBV↑': '✅' if obv_rising else '❌',
            '量比': round(vol_ratio, 2),
            '20日涨幅%': round(returns_20d, 2),
            '评分': score,
        }
    except:
        return None

col_etf1, col_etf2 = st.columns([3, 1])
with col_etf2:
    min_score_etf = st.slider("最低评分", 0, 5, 4, key="etf_min_score")

if st.button("🔍 扫描 ETF 资金流向", type="primary"):
    etf_results = []
    progress = st.progress(0)
    etf_list = list(SECTOR_ETFS.keys())
    
    for i, ticker in enumerate(etf_list):
        progress.progress((i + 1) / len(etf_list))
        data = get_etf_flow_data(ticker)
        if data is not None and not data.empty:
            result = calculate_etf_signals(ticker, data)
            if result:
                etf_results.append(result)
    
    progress.empty()
    
    if etf_results:
        etf_df = pd.DataFrame(etf_results).sort_values('评分', ascending=False)
        st.session_state['etf_scan_results'] = etf_df
        
        # 显示结果
        st.dataframe(etf_df, use_container_width=True, hide_index=True)
        
        # 资金流入板块
        inflow = etf_df[etf_df['评分'] >= min_score_etf]
        if len(inflow) > 0:
            st.success(f"🔥 资金流入板块 ({len(inflow)} 个): " + ", ".join(inflow['板块'].tolist()))
        
        # 弱势板块
        weak = etf_df[etf_df['评分'] <= 2]
        if len(weak) > 0:
            st.warning(f"⚠️ 弱势板块: " + ", ".join(weak['板块'].tolist()))
else:
    if 'etf_scan_results' in st.session_state:
        st.dataframe(st.session_state['etf_scan_results'], use_container_width=True, hide_index=True)

# ============================================================================
# 📤 统一导出到 Claude
# ============================================================================

st.divider()
st.header("📤 导出完整数据到 Claude")

def generate_unified_export():
    """生成统一的导出文本，包含所有模块数据"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    export = f"""# 🦅 宏观战情观察室 - 完整数据快照
生成时间: {timestamp} EST

═══════════════════════════════════════════════════════════════
## 一、流动性指标
═══════════════════════════════════════════════════════════════
- SOFR: {ny_fed['SOFR']:.2f}%
- Repo (TGCR): {ny_fed['TGCR']:.2f}%
- SOFR-Repo 利差: {(ny_fed['SOFR'] - ny_fed['TGCR']):.3f}%
- RRP: ${fed_liq['RRP']:.0f}B (日变化: {fed_liq['RRP_Chg']:.0f}B)
- TGA: ${fed_liq['TGA']:.0f}B (日变化: {fed_liq['TGA_Chg']:.0f}B)
- HYG/LQD: {credit[0]:.3f} (日变化: {credit[1]:.2f}%)

═══════════════════════════════════════════════════════════════
## 二、美债与汇率
═══════════════════════════════════════════════════════════════
- 10Y 收益率: {rates['Yield_10Y']:.2f}%
- 3M 收益率: {rates['Yield_Short']:.2f}%
- 10Y-3M 利差: {rates['Inversion']:.2f}%
- MOVE 指数: {rates['MOVE']:.1f}
- DXY: {rates['DXY']:.2f}
- USDJPY: {rates['USDJPY']:.2f}

═══════════════════════════════════════════════════════════════
## 三、恐慌与情绪指标
═══════════════════════════════════════════════════════════════
- VIX: {vol['VIX']:.2f}
- 币圈恐慌贪婪: {vol['Crypto_Val']} ({vol['Crypto_Text']})
- PCR: {opt['PCR']:.2f}

═══════════════════════════════════════════════════════════════
## 四、交易微观结构
═══════════════════════════════════════════════════════════════
- 期货基差: {deriv['Futures_Basis']:.1f} ({deriv['Basis_Status']})
- Gamma 环境: {deriv['GEX_Net']}
- Vanna 状态: {deriv['Vanna_Status']}
- Put Wall: ${deriv['Put_Wall']:.0f}
- Call Wall: ${deriv['Call_Wall']:.0f}

═══════════════════════════════════════════════════════════════
## 五、GEX 分析
═══════════════════════════════════════════════════════════════
- 当前价格: ${gex_data['spot_price']:.2f}
- 净 GEX: {gex_data['total_gex']:.2f}B
- Gamma Flip Point: ${gex_data['gamma_flip']:.2f}
- Max Pain: ${gex_data['max_pain']:.2f}
- GEX Put Wall: ${gex_data['put_wall']:.2f}
- GEX Call Wall: ${gex_data['call_wall']:.2f}

═══════════════════════════════════════════════════════════════
## 六、规则引擎信号
═══════════════════════════════════════════════════════════════
市场状态: {regime_analysis['regime'].upper()}
综合评分: {regime_analysis['score']:.1f}

关键信号:
"""
    for sig in regime_analysis['signals']:
        export += f"- [{sig['level']}] {sig['msg']}\n"
    
    export += """
═══════════════════════════════════════════════════════════════
## 七、重点新闻
═══════════════════════════════════════════════════════════════
"""
    for item in processed_news[:10]:
        cats = ", ".join(item.get('Categories', ['general']))
        export += f"- [{cats}] {item['Title']} (情绪: {item.get('Sentiment', 'Neutral')})\n"
    
    # 添加 Rotation Score 数据
    export += """
═══════════════════════════════════════════════════════════════
## 八、资金轮动评分 (Rotation Score)
═══════════════════════════════════════════════════════════════
"""
    if 'rotation_results' in st.session_state and st.session_state['rotation_results']:
        rot = st.session_state['rotation_results']
        export += f"**综合评分: {rot['total_score']:.1f}** (-100 到 +100)\n\n"
        for cat_key, cat_data in rot['categories'].items():
            export += f"### {cat_data['name']} (评分: {cat_data['score']:.1f})\n"
            for factor in cat_data['factors']:
                signal = '🟢' if factor['signal'] == 'bullish' else '🔴' if factor['signal'] == 'bearish' else '⚪'
                export += f"- {signal} {factor['desc']}: Z={factor['z_score']:.2f}\n"
    else:
        export += "(请先刷新 Rotation Score 数据)\n"
    
    # 添加 Gamma 输入数据
    export += """
═══════════════════════════════════════════════════════════════
## 九、SpotGamma 数据 (手动输入)
═══════════════════════════════════════════════════════════════
"""
    if 'gamma_qqq_data' in st.session_state and st.session_state['gamma_qqq_data'].get('zero_gamma'):
        g = st.session_state['gamma_qqq_data']
        export += f"**QQQ**\n"
        export += f"- Zero Gamma: ${g.get('zero_gamma')}\n"
        export += f"- Call Wall: ${g.get('call_wall')}\n"
        export += f"- Put Wall: ${g.get('put_wall')}\n"
    else:
        export += "(请在侧边栏输入 SpotGamma 数据)\n"
    
    # 添加 ETF 扫描结果
    export += """
═══════════════════════════════════════════════════════════════
## 十、ETF 资金流向扫描
═══════════════════════════════════════════════════════════════
"""
    if 'etf_scan_results' in st.session_state:
        etf_df = st.session_state['etf_scan_results']
        strong = etf_df[etf_df['评分'] >= 4]
        weak = etf_df[etf_df['评分'] <= 2]
        export += f"**资金流入板块**: {', '.join(strong['板块'].tolist()) if len(strong) > 0 else '无'}\n"
        export += f"**弱势板块**: {', '.join(weak['板块'].tolist()) if len(weak) > 0 else '无'}\n"
    else:
        export += "(请先执行 ETF 扫描)\n"
    
    export += """
═══════════════════════════════════════════════════════════════
## 分析请求
═══════════════════════════════════════════════════════════════
请基于以上完整数据进行深度分析:
1. 当前市场处于什么宏观周期？流动性环境如何？
2. Gamma 环境与资金流向是否共振？
3. 有哪些潜在的风险点需要关注？
4. 今日 QQQ 期权交易的最佳策略是什么？建议的行权价？
5. 哪些板块值得重点关注？
"""
    return export

with st.expander("📋 点击展开完整数据导出", expanded=False):
    st.markdown("""
    <div class="export-box">
    <p>📋 点击下方文本框，全选 (Ctrl+A) 并复制 (Ctrl+C)，然后粘贴给 Claude 进行深度分析</p>
    </div>
    """, unsafe_allow_html=True)
    
    unified_export = generate_unified_export()
    st.text_area("完整数据快照", unified_export, height=500, key="unified_export")

st.divider()
st.caption("🦅 宏观战情观察室 v2.0 | 数据来源: NY Fed, Yahoo Finance, SpotGamma | 仅供参考，不构成投资建议")
