import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import pytz
import feedparser
from transformers import pipeline
from fredapi import Fred

# --- 0. 全局配置 ---
st.set_page_config(page_title="QQQ 宏观战情室", layout="wide", page_icon="🦅")

# 自定义样式: 更加紧凑专业的金融终端风格
st.markdown("""
    <style>
    .metric-card {background-color: #f9f9f9; border-radius: 5px; padding: 10px; border: 1px solid #e0e0e0;}
    .news-card {padding: 10px; margin-bottom: 5px; border-radius: 5px; border-left: 5px solid #ccc;}
    .news-bull {background-color: #e6fffa; border-left-color: #00c04b;}
    .news-bear {background-color: #fff5f5; border-left-color: #ff4b4b;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. 核心模型与数据获取 ---

@st.cache_resource
def load_ai_model():
    """加载 FinBERT AI 模型 (用于新闻情感分析)"""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_data(ttl=3600)
def get_ny_fed_data():
    """获取 SOFR 和 TGCR (Repo) 数据"""
    try:
        url = "https://markets.newyorkfed.org/api/rates/all/latest.json"
        r = requests.get(url, timeout=5).json()
        rates = {'SOFR': 5.3, 'TGCR': 5.3} 
        for item in r.get('refRates', []):
            if item['type'] == 'SOFR': rates['SOFR'] = float(item['percentRate'])
            if item['type'] == 'TGCR': rates['TGCR'] = float(item['percentRate'])
        return rates
    except:
        return {'SOFR': 5.33, 'TGCR': 5.32}

@st.cache_data(ttl=3600)
def get_credit_spreads():
    """计算信贷利差 (HYG/LQD)"""
    try:
        data = yf.download(["HYG", "LQD"], period="5d", progress=False)['Close']
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(0)
        ratio = data['HYG'] / data['LQD']
        current_ratio = ratio.iloc[-1]
        pct_change = ((current_ratio - ratio.iloc[-2]) / ratio.iloc[-2]) * 100
        return current_ratio, pct_change
    except:
        return 0, 0

@st.cache_data(ttl=900)
def get_rates_and_fx():
    """获取美债、汇率、MOVE指数"""
    tickers = ["^IRX", "^TNX", "^TYX", "DX-Y.NYB", "JPY=X", "^MOVE"] 
    res = {}
    try:
        df = yf.download(tickers, period="5d", progress=False)['Close']
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(0)
        
        res['Yield_2Y'] = df.get('^IRX', pd.Series([5.2])).iloc[-1]
        res['Yield_10Y'] = df.get('^TNX', pd.Series([4.2])).iloc[-1]
        res['Yield_30Y'] = df.get('^TYX', pd.Series([4.4])).iloc[-1]
        res['DXY'] = df.get('DX-Y.NYB', pd.Series([104])).iloc[-1]
        res['USDJPY'] = df.get('JPY=X', pd.Series([150])).iloc[-1]
        
        # MOVE 数据容错
        if '^MOVE' in df and not pd.isna(df['^MOVE'].iloc[-1]):
            res['MOVE'] = df['^MOVE'].iloc[-1]
        else:
            res['MOVE'] = 100.0
            
        res['Inversion'] = res['Yield_10Y'] - res['Yield_2Y']
    except:
        res = {'Yield_2Y':5.0, 'Yield_10Y':4.2, 'Yield_30Y':4.3, 'DXY':104, 'USDJPY':150, 'MOVE':100, 'Inversion':-0.8}
    return res

@st.cache_data(ttl=600)
def get_volatility_indices():
    """获取 VIX 和 币圈恐慌指数"""
    data = {}
    # VIX
    try:
        vix = yf.Ticker("^VIX").history(period="2d")['Close'].iloc[-1]
        data['VIX'] = vix
    except: data['VIX'] = 15.0

    # Crypto Fear & Greed
    try:
        r = requests.get("https://api.alternative.me/fng/").json()
        data['Crypto_Val'] = int(r['data'][0]['value'])
        data['Crypto_Text'] = r['data'][0]['value_classification']
    except: 
        data['Crypto_Val'] = 50
        data['Crypto_Text'] = "Unknown"
    return data

@st.cache_data(ttl=600)
def get_qqq_options_data():
    """QQQ 期权 PCR 与 异动雷达"""
    qqq = yf.Ticker("QQQ")
    res = {"PCR": 0.0, "Unusual": []}
    try:
        exp = qqq.options[0]
        chain = qqq.option_chain(exp)
        calls, puts = chain.calls, chain.puts
        
        c_vol = calls['volume'].sum()
        p_vol = puts['volume'].sum()
        if c_vol > 0: res['PCR'] = round(p_vol / c_vol, 2)
        
        unusual = []
        for opt_type, df, icon in [("CALL", calls, "🟢"), ("PUT", puts, "🔴")]:
            hot = df[(df['volume'] > 500) & (df['volume'] > df['openInterest'] * 1.2)]
            for _, row in hot.iterrows():
                unusual.append({
                    "Type": f"{icon} {opt_type}",
                    "Strike": row['strike'],
                    "Vol": int(row['volume']),
                    "OI": int(row['openInterest']),
                    "Ratio": round(row['volume'] / (row['openInterest']+1), 1)
                })
        res['Unusual'] = sorted(unusual, key=lambda x: x['Vol'], reverse=True)[:10]
    except: pass
    return res

@st.cache_data(ttl=3600)
def get_macro_calendar():
    """宏观日历"""
    events = [
        {"Date": "2024-06-12", "Event": "CPI 数据发布", "Type": "Inflation"},
        {"Date": "2024-06-12", "Event": "FOMC 利率决议", "Type": "Fed"},
        {"Date": "2024-06-14", "Event": "BOJ 日本央行会议", "Type": "BOJ"},
        {"Date": "2024-07-05", "Event": "NFP 非农就业", "Type": "Jobs"},
        {"Date": "2024-06-15", "Event": "企业缴税日 (流动性抽取)", "Type": "Liquidity"},
    ]
    today = datetime.date.today()
    upcoming = []
    for e in events:
        d = datetime.datetime.strptime(e['Date'], "%Y-%m-%d").date()
        days = (d - today).days
        if 0 <= days <= 45:
            upcoming.append({**e, "Days": days})
    return sorted(upcoming, key=lambda x: x['Days'])

@st.cache_data(ttl=600)
def get_macro_news():
    """
    [新增功能] 抓取宏观新闻 RSS
    源: CNBC Economy, MarketWatch Top Stories
    """
    feeds = [
        ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("MarketWatch", "http://feeds.marketwatch.com/marketwatch/topstories/"),
        ("WSJ Markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml")
    ]
    
    articles = []
    for source_name, url in feeds:
        try:
            f = feedparser.parse(url)
            # 每个源只取前 4 条，保持速度
            for e in f.entries[:4]:
                articles.append({
                    "Title": e.title,
                    "Link": e.link,
                    "Source": source_name,
                    "Published": e.get('published', 'Today')
                })
        except: pass
    return pd.DataFrame(articles)

# --- 2. 核心算法: 多空评分模型 ---

def calculate_macro_score(ny_fed, credit, rates, vol, opt, news_score_val):
    """
    权重模型 (总分 -10 到 +10)
    """
    score = 0
    details = []
    
    # 1. 流动性 (25%)
    liq_score = 0
    spread = ny_fed['SOFR'] - ny_fed['TGCR']
    if spread > 0.05: 
        liq_score -= 1.5; details.append("🔴 SOFR 异常跳升 (钱紧)")
    elif spread < 0.02:
        liq_score += 0.5
    if credit[1] < -0.5: 
        liq_score -= 1.0; details.append("🔴 信贷利差扩大 (避险)")
    elif credit[1] > 0.2:
        liq_score += 1.0
    score += max(-2.5, min(2.5, liq_score))
    
    # 2. 美债 (25%)
    bond_score = 0
    if rates['Yield_10Y'] > 4.5:
        bond_score -= 1.0; details.append("🔴 10Y 收益率过高")
    elif rates['Yield_10Y'] < 4.0:
        bond_score += 1.0
    if rates['MOVE'] > 110:
        bond_score -= 1.5; details.append("🔴 MOVE 债市恐慌")
    score += max(-2.5, min(2.5, bond_score))
    
    # 3. 恐慌指数 (15%)
    fear_score = 0
    if vol['VIX'] > 25: fear_score -= 1.0; details.append("🔴 VIX 恐慌")
    elif vol['VIX'] < 13: fear_score -= 0.5; details.append("⚠️ VIX 过低")
    else: fear_score += 0.5
    if vol['Crypto_Val'] < 20: fear_score += 0.5; details.append("🟢 币圈恐慌(反弹机会)")
    score += fear_score
    
    # 4. 交易数据 (20%)
    trade_score = 0
    if opt['PCR'] > 1.1: trade_score -= 1.0; details.append("📉 PCR 偏空")
    elif opt['PCR'] < 0.7: trade_score += 1.0; details.append("📈 PCR 偏多")
    
    call_vol = sum([x['Vol'] for x in opt['Unusual'] if "CALL" in x['Type']])
    put_vol = sum([x['Vol'] for x in opt['Unusual'] if "PUT" in x['Type']])
    if call_vol > put_vol * 1.5: trade_score += 1.0
    elif put_vol > call_vol * 1.5: trade_score -= 1.0
    score += max(-2.0, min(2.0, trade_score))
    
    # 5. 新闻 (15%)
    # news_score_val 是归一化的 -1 到 1
    news_contribution = news_score_val * 1.5
    score += news_contribution
    if news_contribution > 0.5: details.append("🟢 宏观新闻偏多")
    if news_contribution < -0.5: details.append("🔴 宏观新闻偏空")
    
    return round(score * (10 / 7.5), 1), details

# --- 3. 界面渲染 (UI) ---

with st.spinner("正在连接全球金融数据源 & 分析新闻情感..."):
    # 1. 加载模型
    ai_model = load_ai_model()
    
    # 2. 获取所有数据
    ny_fed = get_ny_fed_data()
    credit = get_credit_spreads()
    rates = get_rates_and_fx()
    vol = get_volatility_indices()
    opt = get_qqq_options_data()
    cal = get_macro_calendar()
    raw_news = get_macro_news()

    # 3. [新增] 处理新闻情感
    processed_news = []
    sentiment_score_total = 0
    
    if not raw_news.empty:
        # 只取前 10 条进行 AI 分析，防止太慢
        for i, row in raw_news.head(10).iterrows():
            try:
                # AI 分析
                res = ai_model(row['Title'][:512])[0]
                label = res['label']
                score = res['score']
                
                # 映射逻辑
                sentiment = "Neutral"
                val = 0
                if label == 'positive' and score > 0.5:
                    sentiment = "Bullish"
                    val = 1
                elif label == 'negative' and score > 0.5:
                    sentiment = "Bearish"
                    val = -1
                
                sentiment_score_total += val
                processed_news.append({**row, "Sentiment": sentiment})
            except: pass
            
        # 计算平均新闻分 (-1 到 1)
        avg_news_score = sentiment_score_total / max(1, len(processed_news))
    else:
        avg_news_score = 0

    # 4. 计算总分
    final_score, reasons = calculate_macro_score(ny_fed, credit, rates, vol, opt, avg_news_score)

# --- HEADER ---
st.title("🦅 QQQ 宏观战情室 (Macro War Room)")
current_time = datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M EST')
st.caption(f"数据更新时间: {current_time}")

col_score, col_text = st.columns([1, 3])
with col_score:
    color = "red" if final_score < -3 else "green" if final_score > 3 else "gray"
    st.metric("大盘多空综评 (-10 ~ +10)", f"{final_score}", delta_color="off")
    if final_score > 3: st.success("### 偏多 (Bullish)")
    elif final_score < -3: st.error("### 偏空 (Bearish)")
    else: st.info("### 中性震荡 (Neutral)")

with col_text:
    st.markdown("#### 🛡️ 战情综述")
    st.write("主要驱动因子: " + " | ".join(reasons))
    if ny_fed['SOFR'] - ny_fed['TGCR'] > 0.05:
        st.warning("⚠️ 警告: 流动性异常收紧 (SOFR Spike)！")

st.divider()

# --- 模块 1: 流动性 ---
st.subheader("1. 流动性监控 (Liquidity)")
l1, l2, l3, l4 = st.columns(4)
l1.metric("SOFR", f"{ny_fed['SOFR']:.2f}%", f"Spread: {ny_fed['SOFR'] - ny_fed['TGCR']:.3f}")
l2.metric("Repo (TGCR)", f"{ny_fed['TGCR']:.2f}%")
l3.metric("HYG/LQD 比率", f"{credit[0]:.3f}", f"{credit[1]:.2f}% (Risk)")
liq_status = "宽松"
if (ny_fed['SOFR'] - ny_fed['TGCR']) > 0.05: liq_status = "🔴 紧张"
l4.metric("流动性状态", liq_status)

st.divider()

# --- 模块 2: 美债与汇率 ---
st.subheader("2. 美债与汇率 (Rates & FX)")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("10Y 美债收益率", f"{rates['Yield_10Y']:.2f}%")
r2.metric("MOVE (债市恐慌)", f"{rates['MOVE']:.2f}")
r3.metric("2Y/10Y 倒挂", f"{rates['Inversion']:.2f}%")
r4.metric("美元指数 (DXY)", f"{rates['DXY']:.2f}")
r5.metric("美元/日元", f"{rates['USDJPY']:.2f}")

st.divider()

# --- 模块 3: 交易与恐慌 ---
st.subheader("3. 交易数据与恐慌指数 (Trading & Fear)")
t1, t2, t3 = st.columns(3)
t1.metric("QQQ 期权 PCR", f"{opt['PCR']}", "Put/Call Ratio")
t2.metric("VIX 股市恐慌", f"{vol['VIX']:.2f}")
t3.metric("币圈恐慌指数", f"{vol['Crypto_Val']}", f"{vol['Crypto_Text']}")

st.write("**⚡ QQQ 异动雷达 (Unusual Radar)**")
if opt['Unusual']:
    st.dataframe(pd.DataFrame(opt['Unusual']), use_container_width=True)
else:
    st.info("今日暂无显著异动大单。")

st.divider()

# --- [新增] 模块 4: 宏观新闻情报 ---
st.subheader("4. 宏观新闻情报 (AI Sentiment News)")

col_news_list, col_news_stat = st.columns([3, 1])

with col_news_list:
    if processed_news:
        for item in processed_news:
            # 样式映射
            sentiment = item['Sentiment']
            css_class = "news-card"
            icon = "⚪"
            if sentiment == "Bullish": 
                css_class += " news-bull"
                icon = "🟢"
            elif sentiment == "Bearish": 
                css_class += " news-bear"
                icon = "🔴"
            
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{icon} {sentiment}</strong> | <a href="{item['Link']}" target="_blank">{item['Title']}</a>
                <br><span style="font-size:0.8em; color:gray;">Source: {item['Source']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("暂无最新新闻数据。")

with col_news_stat:
    st.markdown("#### AI 情绪统计")
    bulls = len([x for x in processed_news if x['Sentiment'] == 'Bullish'])
    bears = len([x for x in processed_news if x['Sentiment'] == 'Bearish'])
    st.metric("利多新闻数", bulls)
    st.metric("利空新闻数", bears)
    
    if avg_news_score > 0.3:
        st.success("舆情风向: 偏多")
    elif avg_news_score < -0.3:
        st.error("舆情风向: 偏空")
    else:
        st.info("舆情风向: 中性")

st.divider()

# --- 模块 5: 宏观日历 ---
st.subheader("5. 宏观日历 (Macro Calendar)")
c1, c2 = st.columns(2)
with c1:
    if cal:
        for e in cal:
            color = "red" if e['Days'] <= 5 else "black"
            st.markdown(f":{color}[**{e['Date']}**] - {e['Event']} (倒计时: {e['Days']}天)")
    else: st.write("近期无重大事件。")
with c2:
    st.markdown("""
    **FOMC 立场参考**:
    - 🦅 鹰派: Waller (关注通胀)
    - 🕊️ 鸽派: Goolsbee (关注就业)
    """)
