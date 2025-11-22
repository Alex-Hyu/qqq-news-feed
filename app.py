import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import pytz
import feedparser
from transformers import pipeline

# --- 0. 全局配置 ---
st.set_page_config(page_title="QQQ 宏观战情室", layout="wide", page_icon="🦅")

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
    """加载 FinBERT AI 模型"""
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
def get_fed_liquidity():
    """
    [新增] 获取 RRP 和 TGA 数据
    来源: FRED 公开 CSV (无需 API Key)
    """
    res = {"RRP": 0, "RRP_Chg": 0, "TGA": 0, "TGA_Chg": 0}
    try:
        # 1. RRP (逆回购 - 每日) - ID: RRPONTSYD
        rrp_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=RRPONTSYD"
        rrp_df = pd.read_csv(rrp_url)
        res['RRP'] = rrp_df.iloc[-1]['RRPONTSYD'] # 单位: Billions
        res['RRP_Chg'] = res['RRP'] - rrp_df.iloc[-2]['RRPONTSYD']
        
        # 2. TGA (财政部账户 - 周度) - ID: WTREGEN
        # 注: TGA 日度数据很难免费获取，这里使用 FRED 的周度数据作为趋势参考
        tga_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WTREGEN"
        tga_df = pd.read_csv(tga_url)
        res['TGA'] = tga_df.iloc[-1]['WTREGEN'] # 单位: Billions
        res['TGA_Chg'] = res['TGA'] - tga_df.iloc[-2]['WTREGEN']
        
    except Exception as e:
        print(f"FRED Data Error: {e}")
    return res

@st.cache_data(ttl=3600)
def get_credit_spreads():
    """计算信贷利差 (HYG/LQD)"""
    try:
        data = yf.download(["HYG", "LQD"], period="5d", progress=False)['Close']
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(0)
        ratio = data['HYG'] / data['LQD']
        curr = ratio.iloc[-1]
        pct = ((curr - ratio.iloc[-2]) / ratio.iloc[-2]) * 100
        return curr, pct
    except: return 0, 0

@st.cache_data(ttl=900)
def get_rates_and_fx():
    """获取美债、汇率、MOVE"""
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
        res['MOVE'] = df.get('^MOVE', pd.Series([100.0])).iloc[-1]
        res['Inversion'] = res['Yield_10Y'] - res['Yield_2Y']
    except:
        res = {'Yield_2Y':5.0, 'Yield_10Y':4.2, 'Yield_30Y':4.3, 'DXY':104, 'USDJPY':150, 'MOVE':100, 'Inversion':-0.8}
    return res

@st.cache_data(ttl=600)
def get_volatility_indices():
    """VIX & Crypto FNG"""
    data = {}
    try:
        vix = yf.Ticker("^VIX").history(period="2d")['Close'].iloc[-1]
        data['VIX'] = vix
    except: data['VIX'] = 15.0
    try:
        r = requests.get("https://api.alternative.me/fng/").json()
        data['Crypto_Val'] = int(r['data'][0]['value'])
        data['Crypto_Text'] = r['data'][0]['value_classification']
    except: 
        data['Crypto_Val'] = 50; data['Crypto_Text'] = "Unknown"
    return data

@st.cache_data(ttl=600)
def get_qqq_options_data():
    """PCR & Unusual Radar"""
    qqq = yf.Ticker("QQQ")
    res = {"PCR": 0.0, "Unusual": []}
    try:
        exp = qqq.options[0]
        chain = qqq.option_chain(exp)
        calls, puts = chain.calls, chain.puts
        if calls['volume'].sum() > 0: 
            res['PCR'] = round(puts['volume'].sum() / calls['volume'].sum(), 2)
        
        unusual = []
        for opt_type, df, icon in [("CALL", calls, "🟢"), ("PUT", puts, "🔴")]:
            hot = df[(df['volume'] > 500) & (df['volume'] > df['openInterest'] * 1.2)]
            for _, row in hot.iterrows():
                unusual.append({
                    "Type": f"{icon} {opt_type}", "Strike": row['strike'],
                    "Vol": int(row['volume']), "OI": int(row['openInterest']),
                    "Ratio": round(row['volume'] / (row['openInterest']+1), 1)
                })
        res['Unusual'] = sorted(unusual, key=lambda x: x['Vol'], reverse=True)[:10]
    except: pass
    return res

@st.cache_data(ttl=3600)
def get_macro_calendar():
    events = [
        {"Date": "2024-06-12", "Event": "CPI 数据发布", "Type": "Inflation"},
        {"Date": "2024-06-12", "Event": "FOMC 利率决议", "Type": "Fed"},
        {"Date": "2024-06-14", "Event": "BOJ 日本央行会议", "Type": "BOJ"},
        {"Date": "2024-07-05", "Event": "NFP 非农就业", "Type": "Jobs"},
        {"Date": "2024-06-15", "Event": "企业缴税日 (TGA抽水)", "Type": "Liquidity"},
    ]
    today = datetime.date.today()
    upcoming = []
    for e in events:
        d = datetime.datetime.strptime(e['Date'], "%Y-%m-%d").date()
        days = (d - today).days
        if 0 <= days <= 45: upcoming.append({**e, "Days": days})
    return sorted(upcoming, key=lambda x: x['Days'])

@st.cache_data(ttl=600)
def get_macro_news():
    feeds = [
        ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("MarketWatch", "http://feeds.marketwatch.com/marketwatch/topstories/"),
        ("WSJ Markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml")
    ]
    articles = []
    for src, url in feeds:
        try:
            f = feedparser.parse(url)
            for e in f.entries[:4]:
                articles.append({"Title": e.title, "Link": e.link, "Source": src})
        except: pass
    return pd.DataFrame(articles)

# --- 2. 核心算法: 多空评分模型 ---

def calculate_macro_score(ny_fed, fed_liq, credit, rates, vol, opt, news_score_val):
    """
    加入了 RRP 和 TGA 的评分逻辑
    """
    score = 0
    details = []
    
    # --- 1. 流动性 (25%) ---
    liq_score = 0
    
    # A. SOFR Spread
    spread = ny_fed['SOFR'] - ny_fed['TGCR']
    if spread > 0.05: liq_score -= 1.0; details.append("🔴 SOFR 异常跳升 (钱紧)")
    elif spread < 0.02: liq_score += 0.5
    
    # B. 信贷利差
    if credit[1] < -0.5: liq_score -= 0.5; details.append("🔴 信贷利差扩大")
    elif credit[1] > 0.2: liq_score += 0.5
    
    # C. [新增] RRP & TGA (影子流动性)
    # RRP 上升 = 抽水 (Bad), RRP 下降 = 放水 (Good)
    if fed_liq['RRP_Chg'] > 20: # 增加超过200亿
        liq_score -= 0.5; details.append("🔴 RRP 激增 (流动性回收)")
    elif fed_liq['RRP_Chg'] < -20:
        liq_score += 0.5; details.append("🟢 RRP 释放 (流动性释放)")
        
    # TGA 上升 = 抽水 (Bad)
    if fed_liq['TGA_Chg'] > 20:
        liq_score -= 0.5; details.append("🔴 TGA 补库 (财政部抽水)")
    
    score += max(-2.5, min(2.5, liq_score))
    
    # --- 2. 美债 (25%) ---
    bond_score = 0
    if rates['Yield_10Y'] > 4.5: bond_score -= 1.0; details.append("🔴 10Y 收益率过高")
    elif rates['Yield_10Y'] < 4.0: bond_score += 1.0
    if rates['MOVE'] > 110: bond_score -= 1.5; details.append("🔴 MOVE 债市恐慌")
    score += max(-2.5, min(2.5, bond_score))
    
    # --- 3. 恐慌 (15%) ---
    fear_score = 0
    if vol['VIX'] > 25: fear_score -= 1.0; details.append("🔴 VIX 恐慌")
    elif vol['VIX'] < 13: fear_score -= 0.5; details.append("⚠️ VIX 过低")
    if vol['Crypto_Val'] < 20: fear_score += 0.5; details.append("🟢 币圈极度恐慌")
    score += fear_score
    
    # --- 4. 交易 (20%) ---
    trade_score = 0
    if opt['PCR'] > 1.1: trade_score -= 1.0; details.append("📉 PCR 偏空")
    elif opt['PCR'] < 0.7: trade_score += 1.0; details.append("📈 PCR 偏多")
    score += max(-2.0, min(2.0, trade_score))
    
    # --- 5. 新闻 (15%) ---
    news_con = news_score_val * 1.5
    score += news_con
    if news_con < -0.5: details.append("🔴 宏观舆情偏空")
    
    return round(score * (10 / 7.5), 1), details

# --- 3. 界面渲染 (UI) ---

with st.spinner("正在同步美联储、纽联储及全球市场数据..."):
    ai_model = load_ai_model()
    ny_fed = get_ny_fed_data()
    fed_liq = get_fed_liquidity() # 新增
    credit = get_credit_spreads()
    rates = get_rates_and_fx()
    vol = get_volatility_indices()
    opt = get_qqq_options_data()
    cal = get_macro_calendar()
    raw_news = get_macro_news()

    # 新闻 AI 处理
    processed_news = []
    sentiment_total = 0
    if not raw_news.empty:
        for i, row in raw_news.head(8).iterrows():
            try:
                res = ai_model(row['Title'][:512])[0]
                label = res['label']
                score = res['score']
                sent = "Neutral"
                val = 0
                if label == 'positive' and score > 0.5: sent="Bullish"; val=1
                elif label == 'negative' and score > 0.5: sent="Bearish"; val=-1
                sentiment_total += val
                processed_news.append({**row, "Sentiment": sent})
            except: pass
        avg_news_score = sentiment_total / max(1, len(processed_news))
    else: avg_news_score = 0

    final_score, reasons = calculate_macro_score(ny_fed, fed_liq, credit, rates, vol, opt, avg_news_score)

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
    st.write("驱动因子: " + " | ".join(reasons))
    if fed_liq['RRP_Chg'] > 100:
        st.warning("⚠️ 严重警告: RRP 激增，市场流动性正在快速枯竭！")

st.divider()

# --- 模块 1: 流动性 (升级版) ---
st.subheader("1. 流动性监控 (Liquidity)")
l1, l2, l3, l4, l5 = st.columns(5)

l1.metric("SOFR", f"{ny_fed['SOFR']:.2f}%", f"Spread: {ny_fed['SOFR'] - ny_fed['TGCR']:.3f}")
l2.metric("Repo (TGCR)", f"{ny_fed['TGCR']:.2f}%")
# [新增] RRP 和 TGA
l3.metric("RRP (逆回购)", f"${fed_liq['RRP']:.0f}B", f"{fed_liq['RRP_Chg']:.0f}B (变动)", delta_color="inverse")
l4.metric("TGA (财政部)", f"${fed_liq['TGA']:.0f}B", f"{fed_liq['TGA_Chg']:.0f}B (变动)", delta_color="inverse")
# 信贷
l5.metric("HYG/LQD", f"{credit[0]:.3f}", f"{credit[1]:.2f}%")

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

# --- 模块 4: 宏观新闻情报 ---
st.subheader("4. 宏观新闻情报 (AI Sentiment News)")
col_news_list, col_news_stat = st.columns([3, 1])
with col_news_list:
    if processed_news:
        for item in processed_news:
            css_class = "news-card"
            icon = "⚪"
            if item['Sentiment'] == "Bullish": css_class += " news-bull"; icon = "🟢"
            elif item['Sentiment'] == "Bearish": css_class += " news-bear"; icon = "🔴"
            st.markdown(f"""<div class="{css_class}"><strong>{icon} {item['Sentiment']}</strong> | <a href="{item['Link']}" target="_blank">{item['Title']}</a><br><span style="font-size:0.8em;color:gray;">{item['Source']}</span></div>""", unsafe_allow_html=True)
    else: st.write("暂无最新新闻数据。")
with col_news_stat:
    st.metric("新闻情绪分", f"{avg_news_score:.2f}", "(-1 空 ~ 1 多)")

st.divider()

# --- 模块 5: 宏观日历 ---
st.subheader("5. 宏观日历 (Macro Calendar)")
if cal:
    cols = st.columns(len(cal) if len(cal)<5 else 5)
    for idx, e in enumerate(cal[:5]):
        with cols[idx]:
            color = "red" if e['Days'] <= 5 else "black"
            st.markdown(f":{color}[**{e['Event']}**]\n\n{e['Date']} ({e['Days']}天)")
