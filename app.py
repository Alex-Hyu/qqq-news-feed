import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import pytz
import feedparser
from transformers import pipeline
from streamlit_autorefresh import st_autorefresh

# --- 0. 全局配置 ---
st.set_page_config(page_title="QQQ 宏观战情室 Pro", layout="wide", page_icon="🦅")

st.markdown("""
    <style>
    .metric-card {background-color: #f9f9f9; border-radius: 5px; padding: 10px; border: 1px solid #e0e0e0;}
    .news-card {padding: 10px; margin-bottom: 5px; border-radius: 5px; border-left: 5px solid #ccc;}
    .news-bull {background-color: #e6fffa; border-left-color: #00c04b;}
    .news-bear {background-color: #fff5f5; border-left-color: #ff4b4b;}
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 系统状态")
    count = st_autorefresh(interval=30 * 60 * 1000, key="data_refresher")
    st.caption(f"🟢 自动刷新已开启 (30分钟/次)")
    if st.button("🔄 立即手动刷新"):
        st.rerun()

# --- 1. 核心模型与数据获取 ---

@st.cache_resource
def load_ai_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

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
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(0)
        ratio = data['HYG'] / data['LQD']
        curr = ratio.iloc[-1]
        pct = ((curr - ratio.iloc[-2]) / ratio.iloc[-2]) * 100
        return curr, pct
    except: return 0, 0

@st.cache_data(ttl=1800)
def get_rates_and_fx():
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
    except: 
        data['Crypto_Val'] = 50; data['Crypto_Text'] = "Unknown"
    return data

# --- [重点] 自算 Gamma Flip 核心算法 ---
@st.cache_data(ttl=1800)
def get_derivatives_structure():
    """获取 期货基差 + GEX 模型 + 自算 Flip Line"""
    res = {
        "Futures_Basis": 0, "Basis_Status": "Normal", 
        "GEX_Net": "Neutral", "Call_Wall": 0, "Put_Wall": 0, 
        "Flip_Line": 0, "Current_Price": 0,
        "Vanna_Charm_Proxy": "Neutral"
    }
    try:
        # 1. 价格与基差
        market_data = yf.download(["NQ=F", "^NDX", "QQQ"], period="2d", progress=False)['Close']
        if isinstance(market_data.columns, pd.MultiIndex): market_data.columns = market_data.columns.droplevel(0)
        
        fut = market_data['NQ=F'].iloc[-1]
        spot = market_data['^NDX'].iloc[-1]
        qqq_price = market_data['QQQ'].iloc[-1]
        res['Current_Price'] = qqq_price
        
        basis = fut - spot
        res['Futures_Basis'] = basis
        if basis < -10: res['Basis_Status'] = "🔴 Backwardation"
        elif basis > 50: res['Basis_Status'] = "🟢 Contango"
        else: res['Basis_Status'] = "⚪ Flat"
        
        # 2. GEX 计算
        qqq = yf.Ticker("QQQ")
        exp = qqq.options[0] # 最近到期日
        chain = qqq.option_chain(exp)
        calls = chain.calls
        puts = chain.puts
        
        # 2.1 基础墙
        res['Call_Wall'] = calls.loc[calls['openInterest'].idxmax()]['strike']
        res['Put_Wall'] = puts.loc[puts['openInterest'].idxmax()]['strike']
        
        # 2.2 [新增] 估算 Zero Gamma Flip Line
        # 算法: 寻找 Call OI 和 Put OI 累计影响的平衡点
        # 简化模型: Flip Line 往往位于 Put Wall 附近，或 Max Pain 附近
        # 我们使用 "OI 加权中值" 作为近似
        calls['G_Contribution'] = calls['openInterest'] # 正 Gamma 近似
        puts['G_Contribution'] = puts['openInterest'] * -1 # 负 Gamma 近似
        
        # 合并所有 Strike 的 Gamma 贡献
        merged = pd.concat([calls[['strike', 'G_Contribution']], puts[['strike', 'G_Contribution']]])
        gamma_profile = merged.groupby('strike').sum().sort_index()
        
        # 找到由正转负的那个 Strike (Zero Crossing)
        # 通常是从高价(Call主导)跌到低价(Put主导)的过程
        flip_strike = 0
        
        # 简单粗暴法: 找到 Put OI 巨大的那个区域的上方一点点
        # 这里的近似逻辑: 当 Put OI 开始显著大于 Call OI 时，Gamma 转负
        for index, row in gamma_profile.iterrows():
            if row['G_Contribution'] < 0: # 净 Put 主导
                flip_strike = index
                # 往上找第一个 Call 主导的作为边界
                break
        
        # 修正: 如果找不到，就用 Put Wall 作为最强 Flip Line
        if flip_strike == 0:
            res['Flip_Line'] = res['Put_Wall']
        else:
            # 取 Put Wall 和 理论翻转点的均值，平滑数据
            res['Flip_Line'] = (res['Put_Wall'] + flip_strike) / 2
            
        # 强制修正: Flip Line 通常不会离现价太远，如果数据异常，回退到 Put Wall
        if abs(res['Flip_Line'] - qqq_price) > 50:
             res['Flip_Line'] = res['Put_Wall']

        # 3. 判断 GEX 状态
        if qqq_price < res['Flip_Line']:
            res['GEX_Net'] = "🔴 Negative Gamma (高波动)"
        else:
            res['GEX_Net'] = "🟢 Positive Gamma (低波动)"
            
        # 4. Vanna
        if market_data['^NDX'].iloc[-1] > market_data['^NDX'].iloc[-2]:
            res['Vanna_Charm_Proxy'] = "Tailwind (助涨)"
        else:
            res['Vanna_Charm_Proxy'] = "Headwind (阻力)"

    except Exception as e: pass
    return res

@st.cache_data(ttl=1800)
def get_qqq_options_data():
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

@st.cache_data(ttl=1800)
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

# --- 2. 核心算法 ---

def calculate_macro_score(ny_fed, fed_liq, credit, rates, vol, opt, deriv, news_score_val):
    score = 0
    details = []
    
    # 1. 流动性 (25%)
    liq_score = 0
    spread = ny_fed['SOFR'] - ny_fed['TGCR']
    if spread > 0.05: liq_score -= 1.0; details.append("🔴 SOFR 异常")
    elif spread < 0.02: liq_score += 0.5
    if fed_liq['RRP_Chg'] > 20: liq_score -= 0.5; details.append("🔴 RRP 抽水")
    if fed_liq['TGA_Chg'] > 20: liq_score -= 0.5; details.append("🔴 TGA 抽水")
    if credit[1] < -0.5: liq_score -= 0.5
    elif credit[1] > 0.2: liq_score += 0.5
    score += max(-2.5, min(2.5, liq_score))
    
    # 2. 美债 (25%)
    bond_score = 0
    if rates['Yield_10Y'] > 4.5: bond_score -= 1.0
    elif rates['Yield_10Y'] < 4.0: bond_score += 1.0
    if rates['MOVE'] > 110: bond_score -= 1.5
    score += max(-2.5, min(2.5, bond_score))
    
    # 3. 恐慌 (15%)
    fear_score = 0
    if vol['VIX'] > 25: fear_score -= 1.0
    elif vol['VIX'] < 13: fear_score -= 0.5
    if vol['Crypto_Val'] < 20: fear_score += 0.5
    score += fear_score
    
    # 4. 交易与微观结构 (20%)
    trade_score = 0
    if opt['PCR'] > 1.1: trade_score -= 0.5; details.append("📉 PCR 偏空")
    elif opt['PCR'] < 0.7: trade_score += 0.5
    if deriv['Basis_Status'].startswith("🔴"): trade_score -= 1.0; details.append("🔴 期货贴水")
    
    # Flip Line 判定
    if deriv['GEX_Net'].startswith("🔴"):
        trade_score -= 0.5; details.append("🔴 跌破 Gamma Flip (负Gamma)")
    elif deriv['GEX_Net'].startswith("🟢"):
        trade_score += 0.5
        
    score += max(-2.0, min(2.0, trade_score))
    
    # 5. 新闻 (15%)
    news_con = news_score_val * 1.5
    score += news_con
    if news_con < -0.5: details.append("🔴 舆情偏空")
    
    return round(score * (10 / 7.5), 1), details

# --- 3. UI ---

with st.spinner("正在计算 Gamma Flip Line 及同步数据 (30分钟刷新)..."):
    ai_model = load_ai_model()
    ny_fed = get_ny_fed_data()
    fed_liq = get_fed_liquidity()
    credit = get_credit_spreads()
    rates = get_rates_and_fx()
    vol = get_volatility_indices()
    opt = get_qqq_options_data()
    deriv = get_derivatives_structure()
    cal = get_macro_calendar()
    raw_news = get_macro_news()

    processed_news = []
    sentiment_total = 0
    if not raw_news.empty:
        for i, row in raw_news.head(8).iterrows():
            try:
                res = ai_model(row['Title'][:512])[0]
                label = res['label']
                score = res['score']
                sent = "Neutral"; val = 0
                if label == 'positive' and score > 0.5: sent="Bullish"; val=1
                elif label == 'negative' and score > 0.5: sent="Bearish"; val=-1
                sentiment_total += val
                processed_news.append({**row, "Sentiment": sent})
            except: pass
        avg_news_score = sentiment_total / max(1, len(processed_news))
    else: avg_news_score = 0

    final_score, reasons = calculate_macro_score(ny_fed, fed_liq, credit, rates, vol, opt, deriv, avg_news_score)

# --- HEADER ---
st.title("🦅 QQQ 宏观战情室 Pro (Live)")
current_time = datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M EST')
st.caption(f"上次更新: {current_time} | 自动刷新: 开启 (30分钟)")

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

# 1. 流动性
st.subheader("1. 流动性监控 (Liquidity)")
l1, l2, l3, l4, l5 = st.columns(5)
l1.metric("SOFR", f"{ny_fed['SOFR']:.2f}%", f"Spread: {ny_fed['SOFR'] - ny_fed['TGCR']:.3f}")
l2.metric("Repo (TGCR)", f"{ny_fed['TGCR']:.2f}%")
l3.metric("RRP (逆回购)", f"${fed_liq['RRP']:.0f}B", f"{fed_liq['RRP_Chg']:.0f}B", delta_color="inverse")
l4.metric("TGA (财政部)", f"${fed_liq['TGA']:.0f}B", f"{fed_liq['TGA_Chg']:.0f}B", delta_color="inverse")
l5.metric("HYG/LQD", f"{credit[0]:.3f}", f"{credit[1]:.2f}%")

st.divider()

# 2. 美债
st.subheader("2. 美债与汇率 (Rates & FX)")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("10Y 美债收益率", f"{rates['Yield_10Y']:.2f}%")
r2.metric("MOVE (债市恐慌)", f"{rates['MOVE']:.2f}")
r3.metric("2Y/10Y 倒挂", f"{rates['Inversion']:.2f}%")
r4.metric("美元指数 (DXY)", f"{rates['DXY']:.2f}")
r5.metric("美元/日元", f"{rates['USDJPY']:.2f}")

st.divider()

# 3. 交易与微观结构 (含 Flip Line)
st.subheader("3. 交易与微观结构 (Gamma Flip & GEX)")
t1, t2, t3, t4 = st.columns(4)

t1.metric("QQQ 期权 PCR", f"{opt['PCR']}", "Put/Call Ratio")
t2.metric("VIX 股市恐慌", f"{vol['VIX']:.2f}")
t3.metric("币圈恐慌指数", f"{vol['Crypto_Val']}", f"{vol['Crypto_Text']}")
t4.metric("期货基差 (Basis)", f"{deriv['Futures_Basis']:.2f}", deriv['Basis_Status'])

g1, g2, g3 = st.columns(3)
# 这里显示自算的 Flip Line
g1.metric("Gamma Flip Line (自算)", f"${deriv['Flip_Line']:.2f}", deriv['GEX_Net'], delta_color="off")
g2.metric("Put Wall (强支撑)", f"${deriv['Put_Wall']}", "最大空头Gamma聚集")
g3.metric("Call Wall (强阻力)", f"${deriv['Call_Wall']}", "最大多头Gamma聚集")

with st.expander("查看 QQQ 异动雷达与 Vanna/Charm 状态", expanded=True):
    c_ex1, c_ex2 = st.columns([2, 1])
    with c_ex1:
        st.write("**⚡ 异动雷达 (Unusual Volume > OI)**")
        if opt['Unusual']: st.dataframe(pd.DataFrame(opt['Unusual']), use_container_width=True)
        else: st.info("今日无显著异动。")
    with c_ex2:
        st.write("**Greek Flows (Proxy)**")
        st.info(f"🔮 Vanna/Charm 状态: **{deriv['Vanna_Charm_Proxy']}**")
        st.caption("注: 若VIX下跌，Dealer解套Call，形成Vanna助涨；若VIX暴涨则反之。")

st.divider()

# 4. 新闻
st.subheader("4. 宏观新闻情报 (AI Sentiment)")
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

# 5. 日历
st.subheader("5. 宏观日历")
if cal:
    cols = st.columns(len(cal) if len(cal)<5 else 5)
    for idx, e in enumerate(cal[:5]):
        with cols[idx]:
            color = "red" if e['Days'] <= 5 else "black"
            st.markdown(f":{color}[**{e['Event']}**]\n\n{e['Date']} ({e['Days']}天)")
