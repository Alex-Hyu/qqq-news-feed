import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline
import datetime
import feedparser
import requests
import numpy as np

# --- 页面配置 ---
st.set_page_config(page_title="QQQ 宏观流动性雷达", layout="wide", page_icon="🦅")

# --- 缓存区 (模型加载) ---
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# --- 核心功能 1: 获取流动性数据 (纽约联储 API) ---
@st.cache_data(ttl=3600) # 1小时更新一次
def get_liquidity_data():
    """
    从纽约联储获取官方 SOFR 和 TGCR (作为 Repo 代表) 数据
    """
    try:
        # 纽约联储官方公开 API
        url = "https://markets.newyorkfed.org/api/rates/all/latest.json"
        r = requests.get(url)
        data = r.json()
        
        rates = {}
        # 解析数据
        for item in data.get('refRates', []):
            if item['type'] == 'SOFR':
                rates['SOFR'] = float(item['percentRate'])
            if item['type'] == 'TGCR': # Tri-Party General Collateral Rate (Repo 代理)
                rates['TGCR'] = float(item['percentRate'])
                
        # 如果 API 偶尔抽风，给个兜底数据 (基于当前市场利率)
        if 'SOFR' not in rates: rates['SOFR'] = 5.30
        if 'TGCR' not in rates: rates['TGCR'] = 5.30
            
        return rates
    except:
        return {'SOFR': 5.30, 'TGCR': 5.30}

# --- 核心功能 2: 获取恐慌贪婪指数 ---
@st.cache_data(ttl=1800)
def get_fear_greed():
    indices = {}
    
    # 1. 币圈恐慌指数 (API)
    try:
        r = requests.get("https://api.alternative.me/fng/")
        data = r.json()
        indices['Crypto_Value'] = int(data['data'][0]['value'])
        indices['Crypto_Label'] = data['data'][0]['value_classification']
    except:
        indices['Crypto_Value'] = 50
        indices['Crypto_Label'] = "Unknown"

    # 2. 股市恐慌指数 (用 VIX 和 动量 模拟 CNN 指数，因为 CNN 反爬虫严重)
    try:
        market_data = yf.Ticker("^VIX")
        vix = market_data.history(period="1d")['Close'].iloc[-1]
        
        # 简单映射算法: VIX 越高，恐慌越严重 (0-100, 100是极度贪婪)
        # VIX 12 = Greed (80), VIX 30 = Fear (20)
        stock_fng = max(0, min(100, 100 - (vix - 10) * 4)) 
        
        indices['Stock_Value'] = int(stock_fng)
        indices['VIX'] = vix
        
        if stock_fng > 75: indices['Stock_Label'] = "极度贪婪 (Extreme Greed)"
        elif stock_fng > 55: indices['Stock_Label'] = "贪婪 (Greed)"
        elif stock_fng < 25: indices['Stock_Label'] = "极度恐慌 (Extreme Fear)"
        elif stock_fng < 45: indices['Stock_Label'] = "恐慌 (Fear)"
        else: indices['Stock_Label'] = "中性 (Neutral)"
        
    except:
        indices['Stock_Value'] = 50
        indices['Stock_Label'] = "Neutral"
        indices['VIX'] = 0
        
    return indices

# --- 核心功能 3: 综合新闻与价格 ---
@st.cache_data(ttl=300)
def get_market_news_and_price():
    # 获取价格
    tickers = ["QQQ", "^TNX"]
    prices = {}
    try:
        data = yf.download(tickers, period="2d", progress=False)['Close']
        # 数据清洗
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(0)
        
        for t in tickers:
            try:
                prev = data[t].iloc[-2]
                curr = data[t].iloc[-1]
                prices[t] = ((curr - prev) / prev) * 100
            except: prices[t] = 0.0
    except:
        prices = {"QQQ": 0.0, "^TNX": 0.0}

    # 获取新闻
    all_news = []
    
    # RSS 源
    feeds = [
        ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("MarketWatch", "http://feeds.marketwatch.com/marketwatch/topstories/"),
        ("WSJ Business", "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml")
    ]
    
    for name, url in feeds:
        try:
            f = feedparser.parse(url)
            for e in f.entries[:4]: # 每个源取前4条
                all_news.append({
                    "Title": e.title,
                    "Link": e.link,
                    "Source": name,
                    "Time": datetime.datetime.now()
                })
        except: pass
        
    return prices, pd.DataFrame(all_news)

# --- 逻辑判断引擎 ---
def analyze_macro_context(sofr, tgcr, stock_fng, news_score):
    """
    上帝视角算法：结合流动性、情绪、新闻给出最终判断
    """
    # 1. 流动性判断
    # 当前基准利率约为 5.3% (假设). 如果 SOFR 飙升远超 TGCR，说明钱很贵
    spread = sofr - tgcr
    liquidity_status = "中性 (Neutral)"
    liquidity_score = 0 # -1 (紧), 0 (中), 1 (松)
    
    if sofr > 5.40 or spread > 0.10: 
        liquidity_status = "🔴 紧张 (Tight/Stress)"
        liquidity_score = -1
    elif sofr < 5.20:
        liquidity_status = "🟢 宽松 (Loose)"
        liquidity_score = 1
    else:
        liquidity_status = "⚪ 平稳 (Stable)"
        liquidity_score = 0
        
    # 2. 最终宏观趋势判断
    # 逻辑：流动性紧张 = 无论情绪如何都偏空
    # 逻辑：流动性平稳 + 极度恐慌 = 抄底机会 (Bullish)
    # 逻辑：流动性平稳 + 极度贪婪 = 见顶风险 (Bearish)
    
    verdict = "中性震荡 (Neutral)"
    verdict_color = "gray"
    explanation = "市场处于平衡状态，关注特定个股新闻。"
    
    if liquidity_score == -1:
        verdict = "空头趋势 (Bearish)"
        verdict_color = "red"
        explanation = "警告：流动性出现紧张迹象 (SOFR/Repo 异常)。此时应现金为王，避免高风险资产。"
    
    elif stock_fng < 20 and news_score > -0.5:
        verdict = "超卖反弹 (Rebound Long)"
        verdict_color = "green"
        explanation = "市场极度恐慌，但基本面新闻未全面崩盘，存在反弹机会。"
        
    elif stock_fng > 80:
        verdict = "过热预警 (Overheated)"
        verdict_color = "orange"
        explanation = "市场极度贪婪，随时可能回调。建议止盈或对冲。"
        
    elif news_score > 0.5 and liquidity_score >= 0:
        verdict = "多头趋势 (Bullish)"
        verdict_color = "green"
        explanation = "宏观新闻向好，且流动性充裕，利好 QQQ。"
        
    return liquidity_status, verdict, verdict_color, explanation

# --- UI 渲染 ---
st.title("🦅 QQQ 宏观全景雷达")
st.markdown("集成 **SOFR 流动性** | **恐慌指数** | **AI 新闻分析** 的三位一体决策系统")

with st.spinner("正在连接美联储与市场数据源..."):
    liq_data = get_liquidity_data()
    fng_data = get_fear_greed()
    prices, df_news = get_market_news_and_price()
    sentiment_pipe = load_sentiment_model()

# --- 第一部分：宏观仪表盘 ---
st.subheader("1. 宏观仪表盘 (Macro Dashboard)")

col1, col2, col3, col4 = st.columns(4)

# SOFR 展示
sofr_val = liq_data.get('SOFR', 0)
col1.metric("SOFR (资金成本)", f"{sofr_val:.2f}%", "纽约联储基准")

# GC Repo 展示 (使用 TGCR)
tgcr_val = liq_data.get('TGCR', 0)
col2.metric("Repo/TGCR (回购利率)", f"{tgcr_val:.2f}%", f"Spread: {sofr_val - tgcr_val:.2f}")

# 股市情绪
s_val = fng_data.get('Stock_Value', 50)
s_label = fng_data.get('Stock_Label', 'Neutral')
col3.metric("美股情绪", f"{s_val}/100", s_label, delta_color="off")

# 币圈情绪
c_val = fng_data.get('Crypto_Value', 50)
c_label = fng_data.get('Crypto_Label', 'Unknown')
col4.metric("加密货币情绪", f"{c_val}/100", c_label, delta_color="off")

st.divider()

# --- 第二部分：AI 新闻处理与最终判断 ---
st.subheader("2. 智能研判 (Smart Verdict)")

# 处理新闻情绪
bull_count = 0
bear_count = 0
news_score_agg = 0 # -1 到 1

if not df_news.empty:
    # 只取前 10 条分析以节省时间
    process_df = df_news.head(10).copy()
    results = []
    
    # 进度条
    bar = st.progress(0, "AI 正在阅读新闻...")
    
    for i, row in process_df.iterrows():
        try:
            out = sentiment_pipe(row['Title'][:512])[0]
            label = out['label']
            
            # 简单的 QQQ 逻辑映射
            impact = "中性"
            headline = row['Title'].lower()
            
            if label == 'positive': 
                impact = "利多 (Bullish)"
                bull_count += 1
                news_score_agg += 1
            elif label == 'negative': 
                impact = "利空 (Bearish)"
                bear_count += 1
                news_score_agg -= 1
                
            # 特殊关键词覆盖
            if "inflation" in headline and "rise" in headline: 
                impact = "重大利空 (Inflation)"
                bear_count += 1
            if "rate cut" in headline:
                impact = "重大利多 (Rate Cut)"
                bull_count += 2
                
            results.append({**row, "AI_Signal": impact})
        except: pass
        bar.progress((i+1)/10)
    bar.empty()
    
    # 归一化新闻分数
    total_scanned = bull_count + bear_count + 1
    final_news_score = news_score_agg / total_scanned 
    
    # 调用核心判断逻辑
    liq_status, final_verdict, v_color, reason = analyze_macro_context(
        sofr_val, tgcr_val, s_val, final_news_score
    )
    
    # 展示最终大结论
    c1, c2 = st.columns([1, 2])
    with c1:
        st.info(f"流动性状态: **{liq_status}**")
    with c2:
        if v_color == "red": st.error(f"当前趋势判断: **{final_verdict}**")
        elif v_color == "green": st.success(f"当前趋势判断: **{final_verdict}**")
        else: st.warning(f"当前趋势判断: **{final_verdict}**")
        
    st.caption(f"🔎 判词: {reason}")

    # 展示新闻列表
    with st.expander("查看详细新闻源分析", expanded=True):
        res_df = pd.DataFrame(results)
        for i, row in res_df.iterrows():
            icon = "🟢" if "利多" in row['AI_Signal'] else "🔴" if "利空" in row['AI_Signal'] else "⚪"
            st.write(f"{icon} **{row['AI_Signal']}** | [{row['Title']}]({row['Link']})")
            st.caption(f"来源: {row['Source']}")
