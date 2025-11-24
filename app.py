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

# --- [侧边栏] 配置与刷新 ---
with st.sidebar:
    st.header("⚙️ 设置")
    
    # [修改] 这里已经填入了你的 API Key，默认隐藏显示
    av_api_key = st.text_input(
        "AlphaVantage API Key", 
        value="UMWB63OXOOCIZHXR", 
        type="password", 
        help="用于获取真实宏观日历数据"
    )
    
    st.divider()
    
    st.subheader("系统状态")
    # 30分钟自动刷新
    count = st_autorefresh(interval=30 * 60 * 1000, key="data_refresher")
    st.caption(f"🟢 自动刷新: 开启 (30分钟)")
    if st.button("🔄 立即刷新"):
        st.rerun()

# --- 1. 核心模型与数据获取 ---

@st.cache_resource
def load_ai_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# 宏观数据
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

# RRP/TGA
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

# 市场数据
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

# GEX/Flip Line
@st.cache_data(ttl=1800)
def get_derivatives_structure():
    res = {
        "Futures_Basis": 0, "Basis_Status": "Normal", 
        "GEX_Net": "Neutral", "Call_Wall": 0, "Put_Wall": 0, 
        "Flip_Line": 0, "Current_Price": 0,
        "Vanna_Charm_Proxy": "Neutral"
    }
    try:
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
        
        qqq = yf.Ticker("QQQ")
        exp = qqq.options[0]
        chain = qqq.option_chain(exp)
        calls = chain.calls
        puts = chain.puts
        
        res['Call_Wall'] = calls.loc[calls['openInterest'].idxmax()]['strike']
        res['Put_Wall'] = puts.loc[puts['openInterest'].idxmax()]['strike']
        
        calls['G_Contribution'] = calls['openInterest']
        puts['G_Contribution'] = puts['openInterest'] * -1
        merged = pd.concat([calls[['strike', 'G_Contribution']], puts[['strike', 'G_Contribution']]])
        gamma_profile = merged.groupby('strike').sum().sort_index()
        
        flip_strike = 0
        for index, row in gamma_profile.iterrows():
            if row['G_Contribution'] < 0:
                flip_strike = index
                break
        
        if flip_strike == 0: res['Flip_Line'] = res['Put_Wall']
        else: res['Flip_Line'] = (res['Put_Wall'] + flip_strike) / 2
        
        if abs(res['Flip_Line'] - qqq_price) > 50: res['Flip_Line'] = res['Put_Wall']
        if qqq_price < res['Flip_Line']: res['GEX_Net'] = "🔴 Negative Gamma"
        else: res['GEX_Net'] = "🟢 Positive Gamma"
            
        if market_data['^NDX'].iloc[-1] > market_data['^NDX'].iloc[-2]:
            res['Vanna_Charm_Proxy'] = "Tailwind (助涨)"
        else: res['Vanna_Charm_Proxy'] = "Headwind (阻力)"
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

# --- 双重保障的宏观日历 ---
@st.cache_data(ttl=3600)
def get_macro_calendar(api_key=""):
    """
    优先使用 Alpha Vantage API (Key已内置)
    失败则使用算法估算
    """
    # 方案 A: API 模式
    if api_key:
        try:
            url = f"https://www.alphavantage.co/query?function=ECONOMIC_CALENDAR&apikey={api_key}"
            r = requests.get(url, timeout=5)
            df = pd.read_csv(StringIO(r.text))
            
            # 过滤美元数据
            df = df[df['currency'] == 'USD']
            
            # 智能筛选关键词
            keywords = ["GDP", "Unemployment", "CPI", "Interest Rate", "Payroll", "FOMC", "PCE", "Inventories"]
            df['is_important'] = df['event'].apply(lambda x: any(k in x for k in keywords))
            df = df[df['is_important']]
            
            # 只要未来的
            today = datetime.date.today().strftime("%Y-%m-%d")
            df = df[df['date'] >= today].sort_values('date').head(10)
            
            display_df = df[['date', 'time', 'event', 'estimate', 'previous']].copy()
            display_df.columns = ['Date', 'Time', 'Event', 'Est', 'Prev']
            
            # 如果没数据 (比如周末或假期)，可能返回空，这时触发方案 B
            if not display_df.empty:
                return display_df, "API Data (AlphaVantage)"
            
        except Exception as e:
            pass # 失败则静默进入方案 B

    # 方案 B: 算法估算兜底
    today = datetime.date.today()
    events = []
    
    # 估算 CPI (每月12号左右)
    next_month = today.replace(day=28) + datetime.timedelta(days=4)
    next_cpi = today.replace(day=12) 
    if today.day > 12: next_cpi = (next_month - datetime.timedelta(days=1)).replace(day=12)
    events.append({"Date": next_cpi, "Event": "CPI 通胀数据 (估算)", "Type": "Inflation"})
    
    # 估算 非农 (每月5号左右)
    next_nfp = today.replace(day=5)
    if today.day > 5: next_nfp = (next_month - datetime.timedelta(days=1)).replace(day=5)
    events.append({"Date": next_nfp, "Event": "Nonfarm Payrolls (估算)", "Type": "Jobs"})
    
    # 估算 FOMC
    known_fomc = ["2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-12-10"]
    for d_str in known_fomc:
        d = datetime.datetime.strptime(d_str, "%Y-%m-%d").date()
        if d >= today:
            events.append({"Date": d, "Event": "FOMC 利率决议 (预设)", "Type": "Fed"})
            break 
            
    events.append({"Date": datetime.date(today.year, 6, 15), "Event": "Q2 缴税日 (流动性抽水)", "Type": "Liquidity"})
    
    events = sorted(events, key=lambda x: x['Date'])
    df = pd.DataFrame(events)
    df = df[df['Date'] >= today].head(5)
    
    display_df = df.copy()
    display_df['Time'] = "N/A"
    display_df['Est'] = "--"
    display_df['Prev'] = "--"
    display_df = display_df[['Date', 'Time', 'Event', 'Est', 'Prev']]
    
    return display_df, "备用数据 (Estimated)"

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
    if deriv['GEX_Net'].startswith("🔴"): trade_score -= 0.5; details.append("🔴 跌破 Gamma Flip")
    elif deriv['GEX_Net'].startswith("🟢"): trade_score += 0.5
    score += max(-2.0, min(2.0, trade_score))
    
    # 5. 新闻 (15%)
    news_con = news_score_val * 1.5
    score += news_con
    if news_con < -0.5: details.append("🔴 舆情偏空")
    
    return round(score * (10 / 7.5), 1), details

# --- 3. UI ---

with st.spinner("正在同步全球市场数据 (30分钟刷新)..."):
    ai_model = load_ai_model()
    ny_fed = get_ny_fed_data()
    fed_liq = get_fed_liquidity()
    credit = get_credit_spreads()
    rates = get_rates_and_fx()
    vol = get_volatility_indices()
    opt = get_qqq_options_data()
    deriv = get_derivatives_structure()
    # 传入 API Key
    cal_df, cal_source = get_macro_calendar(av_api_key)
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

# 3. 交易与微观结构
st.subheader("3. 交易与微观结构 (Gamma Flip & GEX)")
t1, t2, t3, t4 = st.columns(4)

t1.metric("QQQ 期权 PCR", f"{opt['PCR']}", "Put/Call Ratio")
t2.metric("VIX 股市恐慌", f"{vol['VIX']:.2f}")
t3.metric("币圈恐慌指数", f"{vol['Crypto_Val']}", f"{vol['Crypto_Text']}")
t4.metric("期货基差 (Basis)", f"{deriv['Futures_Basis']:.2f}", deriv['Basis_Status'])

g1, g2, g3 = st.columns(3)
g1.metric("Gamma Flip Line (自算)", f"${deriv['Flip_Line']:.2f}", deriv['GEX_Net'], delta_color="off")
g2.metric("Put Wall (强支撑)", f"${deriv['Put_Wall']}", "最大空头Gamma")
g3.metric("Call Wall (强阻力)", f"${deriv['Call_Wall']}", "最大多头Gamma")

with st.expander("📚 交易员参考手册：如何解读 PCR (OI)？", expanded=False):
    st.markdown("""
    #### 1. 数值 > 1.2 (高位 - 极度悲观)
    *   **直观感觉**: 大家都看空。做市商手里全是 Short Put (Long Delta)。
    *   **🛡️ 操作**: 只要 QQQ 没崩，意味着底部支撑强。反弹时做市商必须买回对冲。**反向做多信号。**
    #### 2. 数值 < 0.7 (低位 - 极度贪婪)
    *   **直观感觉**: 大家都看多。做市商手里全是 Short Call (Short Delta)。
    *   **⚠️ 操作**: 上涨吃力 (Call Wall 阻力)。**反向做空/止盈信号。**
    """)

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
st.subheader(f"5. 宏观日历 ({cal_source})")
c1, c2 = st.columns([3, 1])
with c1:
    if not cal_df.empty:
        st.dataframe(
            cal_df,
            column_config={
                "Date": "日期", "Time": "时间", "Event": "事件",
                "Est": "预期", "Prev": "前值"
            },
            hide_index=True, use_container_width=True
        )
    else: st.write("近期无重要数据。")

with c2:
    st.markdown("""
    **Fed 观察**:
    - 🦅 **鹰派**: Waller
    - 🕊️ **鸽派**: Goolsbee
    - ⚖️ **中性**: Powell
    """)
    # ... (上面所有原有代码保持不变) ...

# --- [新增] 模块 6: 日内战术面板 (Intraday Tactical) ---
st.subheader("6. 日内交易战术面板 (0DTE & Micro Structure)")

@st.cache_data(ttl=60) # 1分钟刷新，日内要求高时效
def get_intraday_tactics():
    res = {
        "VWAP": 0, "Price": 0, "Trend": "Neutral",
        "Exp_Move": 0, "Upper_Band": 0, "Lower_Band": 0,
        "0DTE_Call_Vol": 0, "0DTE_Put_Vol": 0, "0DTE_Sentiment": "Neutral"
    }
    try:
        # 1. 获取 QQQ 日内 1分钟 数据计算 VWAP
        # 注意: yfinance 免费版日内数据可能延迟，实盘请以此为参考趋势
        df = yf.download("QQQ", period="1d", interval="5m", progress=False)
        if not df.empty:
            # 计算 VWAP = Cumulative(Price * Vol) / Cumulative(Vol)
            df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['PV'] = df['TP'] * df['Volume']
            vwap = df['PV'].sum() / df['Volume'].sum()
            
            current_price = df['Close'].iloc[-1]
            res['VWAP'] = vwap
            res['Price'] = current_price
            
            if current_price > vwap * 1.001: res['Trend'] = "🟢 多头控盘 (Above VWAP)"
            elif current_price < vwap * 0.999: res['Trend'] = "🔴 空头控盘 (Below VWAP)"
            else: res['Trend'] = "⚪ 震荡 (At VWAP)"
            
        # 2. 计算今日预期波动 (Expected Move)
        # 简化公式: 0DTE ATM Straddle Price (Call + Put)
        # 这里用 VIX 倒推: Exp Move = Price * (VIX/16) * sqrt(1/252)
        # VIX/16 近似日波动率
        vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        daily_vol = (vix / 100) / np.sqrt(252)
        exp_move = res['Price'] * daily_vol
        
        res['Exp_Move'] = exp_move
        res['Upper_Band'] = res['Price'] + exp_move
        res['Lower_Band'] = res['Price'] - exp_move
        
        # 3. 0DTE 情绪 (近似)
        qqq = yf.Ticker("QQQ")
        # 找最近的过期日
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        dates = qqq.options
        target_date = dates[0] # 最近的一期，可能是今天或明天
        
        chain = qqq.option_chain(target_date)
        c_vol = chain.calls['volume'].sum()
        p_vol = chain.puts['volume'].sum()
        
        res['0DTE_Call_Vol'] = c_vol
        res['0DTE_Put_Vol'] = p_vol
        
        if c_vol > p_vol: res['0DTE_Sentiment'] = "🟢 Call 主导 (追涨)"
        else: res['0DTE_Sentiment'] = "🔴 Put 主导 (杀跌/避险)"
        
        res['Expiry_Date'] = target_date

    except Exception as e: pass
    return res

# UI 渲染
with st.spinner("正在计算日内 VWAP 与 0DTE 数据..."):
    tactics = get_intraday_tactics()

c_day1, c_day2, c_day3, c_day4 = st.columns(4)

# 1. VWAP 趋势
c_day1.metric("日内趋势 (VWAP)", f"${tactics['VWAP']:.2f}", tactics['Trend'], delta_color="off")

# 2. 预期波动
c_day2.metric("今日预期波动", f"±${tactics['Exp_Move']:.2f}", f"VIX推算")

# 3. 0DTE 情绪
c_day3.metric(f"短期期权 ({tactics.get('Expiry_Date','')})", tactics['0DTE_Sentiment'], f"C/P Vol: {int(tactics['0DTE_Call_Vol']/1000)}k / {int(tactics['0DTE_Put_Vol']/1000)}k")

# 4. 交易区间
c_day4.metric("今日安全边界", f"${tactics['Lower_Band']:.2f} - ${tactics['Upper_Band']:.2f}", "超跌/超买区域")

# 交易建议展示
with st.expander("🏹 日内期权狙击指南 (Intraday Cheat Sheet)", expanded=True):
    st.markdown(f"""
    *   **当前价格**: `${tactics['Price']:.2f}` vs **VWAP**: `${tactics['VWAP']:.2f}`
    *   **策略**:
        *   若价格 > VWAP 且 Gamma Positive (🟢): **逢低做多 (Buy Calls on Dips)**.
        *   若价格 < VWAP 且 Gamma Negative (🔴): **逢高做空 (Buy Puts on Rallies)**.
        *   若价格触及 `${tactics['Upper_Band']:.2f}` (上轨): 考虑 **反向做空/止盈 (Fade the move)**.
        *   若价格触及 `${tactics['Lower_Band']:.2f}` (下轨): 考虑 **反向做多/止盈 (Buy the dip)**.
    """)
