import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline
import datetime
from datetime import date
import feedparser
import requests
import numpy as np
import pytz

# --- 0. 全局配置 ---
st.set_page_config(page_title="QQQ 宏观战情室", layout="wide", page_icon="🦅")
st.markdown("""
    <style>
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. 数据加载模块 ---

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_data(ttl=3600)
def get_macro_liquidity():
    """获取 SOFR 和 TGCR (Repo)"""
    try:
        url = "https://markets.newyorkfed.org/api/rates/all/latest.json"
        r = requests.get(url, timeout=5)
        data = r.json()
        rates = {}
        for item in data.get('refRates', []):
            if item['type'] == 'SOFR': rates['SOFR'] = float(item['percentRate'])
            if item['type'] == 'TGCR': rates['TGCR'] = float(item['percentRate'])
        
        if 'SOFR' not in rates: rates['SOFR'] = 5.30
        if 'TGCR' not in rates: rates['TGCR'] = 5.30
        
        spread = rates['SOFR'] - rates['TGCR']
        rates['Spread'] = spread
        
        if spread > 0.10: rates['Status'] = "🔴 紧张 (Stress)"
        elif spread > 0.05: rates['Status'] = "🟠 偏紧 (Tight)"
        elif rates['SOFR'] < 4.0: rates['Status'] = "🟢 极度宽松 (Loose)"
        else: rates['Status'] = "⚪ 平稳 (Neutral)"
            
        return rates
    except:
        return {'SOFR': 5.30, 'TGCR': 5.30, 'Spread': 0, 'Status': "数据暂缺"}

@st.cache_data(ttl=600)
def get_market_sentiment_data():
    """获取 VIX, QQQ, 币圈恐慌, 股市恐慌"""
    res = {}
    # Crypto
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=3)
        d = r.json()
        res['Crypto_FNG'] = int(d['data'][0]['value'])
        res['Crypto_Text'] = d['data'][0]['value_classification']
    except:
        res['Crypto_FNG'] = 50; res['Crypto_Text'] = "Unknown"

    # Stock (VIX + QQQ)
    try:
        df = yf.download(["^VIX", "QQQ"], period="5d", progress=False)['Close']
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(0)
        
        res['VIX'] = df['^VIX'].iloc[-1]
        res['VIX_Chg'] = df['^VIX'].iloc[-1] - df['^VIX'].iloc[-2]
        res['QQQ_Price'] = df['QQQ'].iloc[-1]
        res['QQQ_Pct'] = ((df['QQQ'].iloc[-1] - df['QQQ'].iloc[-2]) / df['QQQ'].iloc[-2]) * 100
        
        # 模拟股市恐慌指数
        vix_score = max(0, min(100, 100 - (res['VIX'] - 10) * 4))
        ma5 = df['QQQ'].mean()
        mom_score = 70 if res['QQQ_Price'] > ma5 else 30
        final_stock_fng = int((vix_score * 0.6) + (mom_score * 0.4))
        
        res['Stock_FNG'] = final_stock_fng
        if final_stock_fng > 75: res['Stock_Text'] = "极度贪婪"
        elif final_stock_fng > 55: res['Stock_Text'] = "贪婪"
        elif final_stock_fng < 25: res['Stock_Text'] = "极度恐慌"
        elif final_stock_fng < 45: res['Stock_Text'] = "恐慌"
        else: res['Stock_Text'] = "中性"
    except:
        res['VIX'] = 0; res['VIX_Chg'] = 0; res['Stock_FNG'] = 50; res['Stock_Text'] = "Unknown"
        res['QQQ_Price'] = 0; res['QQQ_Pct'] = 0
        
    return res

@st.cache_data(ttl=900)
def get_options_radar():
    """QQQ 期权链分析"""
    qqq = yf.Ticker("QQQ")
    data = {"PCR": 0, "Sentiment": "Neutral", "Unusual": []}
    try:
        exps = qqq.options[:1]
        call_vol, put_vol = 0, 0
        unusual_list = []
        
        for date in exps:
            chain = qqq.option_chain(date)
            calls = chain.calls
            puts = chain.puts
            call_vol += calls['volume'].sum()
            put_vol += puts['volume'].sum()
            
            # 异动: Vol > 1000 且 > 1.5倍 OI
            hot_calls = calls[(calls['volume']>1000) & (calls['volume'] > calls['openInterest']*1.5)]
            for _, r in hot_calls.iterrows():
                unusual_list.append({"Type": "CALL 🟢", "Strike": r['strike'], "Exp": date, "Vol": r['volume'], "OI": r['openInterest'], "Ratio": round(r['volume']/(r['openInterest']+1), 1)})
            
            hot_puts = puts[(puts['volume']>1000) & (puts['volume'] > puts['openInterest']*1.5)]
            for _, r in hot_puts.iterrows():
                unusual_list.append({"Type": "PUT 🔴", "Strike": r['strike'], "Exp": date, "Vol": r['volume'], "OI": r['openInterest'], "Ratio": round(r['volume']/(r['openInterest']+1), 1)})
                
        if call_vol > 0:
            pcr = put_vol / call_vol
            data['PCR'] = round(pcr, 2)
            if pcr > 1.2: data['Sentiment'] = "看空/对冲"
            elif pcr < 0.7: data['Sentiment'] = "极度看多"
            else: data['Sentiment'] = "中性"
            
        data['Unusual'] = sorted(unusual_list, key=lambda x: x['Vol'], reverse=True)[:10]
    except: pass
    return data

@st.cache_data(ttl=3600)
def get_calendar_events():
    """
    功能: 智能获取下一次 CPI, 非农, FOMC 日期
    (基于 2024-2025 官方预定表)
    """
    # 手动维护的官方日程表 (End 2024 - Early 2025)
    # 格式: YYYY-MM-DD
    schedule = [
        # --- 2024 ---
        {"Event": "📊 非农就业 (NFP)", "Date": "2024-11-01"},
        {"Event": "🏛️ FOMC 利率决议", "Date": "2024-11-07"},
        {"Event": "📈 CPI 通胀数据", "Date": "2024-11-13"},
        {"Event": "📊 非农就业 (NFP)", "Date": "2024-12-06"},
        {"Event": "📈 CPI 通胀数据", "Date": "2024-12-11"},
        {"Event": "🏛️ FOMC 利率决议", "Date": "2024-12-18"},
        
        # --- 2025 ---
        {"Event": "📊 非农就业 (NFP)", "Date": "2025-01-03"},
        {"Event": "📈 CPI 通胀数据", "Date": "2025-01-10"}, # 预估
        {"Event": "🏛️ FOMC 利率决议", "Date": "2025-01-29"},
        {"Event": "📊 非农就业 (NFP)", "Date": "2025-02-07"},
        {"Event": "📈 CPI 通胀数据", "Date": "2025-02-12"}, # 预估
        {"Event": "🏛️ FOMC 利率决议", "Date": "2025-03-19"},
    ]
    
    today = datetime.date.today()
    upcoming = []
    
    for e in schedule:
        try:
            e_date = datetime.datetime.strptime(e['Date'], "%Y-%m-%d").date()
            days_left = (e_date - today).days
            # 只显示未来 0 到 90 天内的事件
            if 0 <= days_left <= 90:
                upcoming.append({
                    "Event": e['Event'],
                    "Date": e['Date'],
                    "Days": days_left,
                    "Urgency": "high" if days_left <= 3 else "low"
                })
        except: continue
            
    # 按时间排序
    return sorted(upcoming, key=lambda x: x['Days'])

@st.cache_data(ttl=300)
def get_news_analysis():
    feeds = [
        ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("MarketWatch", "http://feeds.marketwatch.com/marketwatch/topstories/"),
        ("WSJ Business", "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml")
    ]
    articles = []
    for name, url in feeds:
        try:
            f = feedparser.parse(url)
            for e in f.entries[:3]:
                articles.append({"Title": e.title, "Link": e.link, "Source": name})
        except: pass
    return pd.DataFrame(articles)

# --- 2. 核心研判逻辑 ---
def analyze_verdict(liquidity, market_data, options_data):
    score = 0
    reasons = []
    
    # 流动性
    if liquidity['Status'].startswith("🔴"): score -= 3; reasons.append("❌ 流动性危机预警 (SOFR高)")
    elif liquidity['Status'].startswith("🟢"): score += 1; reasons.append("✅ 资金面宽松")
        
    # VIX
    if market_data['VIX'] > 28: score -= 2; reasons.append("❌ 市场恐慌 (VIX爆表)")
    elif market_data['VIX'] < 12: score -= 1; reasons.append("⚠️ 市场自满 (反向指标)")
    elif market_data['Stock_FNG'] < 20: score += 2; reasons.append("✅ 极度恐慌超卖反弹")
        
    # 期权 PCR
    pcr = options_data['PCR']
    if pcr > 1.1: score -= 1; reasons.append("📉 期权对冲保护 (High PCR)")
    elif pcr < 0.6: score += 1; reasons.append("📈 交易员极度看涨 (Low PCR)")
        
    # 趋势
    if market_data['QQQ_Pct'] < -1.5: reasons.append("📉 大盘今日重挫")
    if market_data['QQQ_Pct'] > 1.5: reasons.append("📈 大盘今日强势")
    
    # 结论
    if score >= 2: return "偏多 (Bullish)", "green", reasons
    elif score >= 4: return "强力做多 (Strong Buy)", "green", reasons
    elif score <= -2: return "偏空 (Bearish)", "red", reasons
    elif score <= -4: return "强力做空 (Strong Sell)", "red", reasons
    else: return "中性震荡 (Neutral)", "gray", reasons

# --- 3. 界面渲染 ---
with st.spinner("正在连接全球数据源..."):
    liq = get_macro_liquidity()
    mkt = get_market_sentiment_data()
    opt = get_options_radar()
    cal = get_calendar_events()
    news_df = get_news_analysis()
    sentiment_model = load_sentiment_model()

st.title("🦅 QQQ 宏观战情室 (Macro War Room)")
st.caption(f"Last Update: {datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%H:%M EST')}")

# 模块 A: 研判
verdict, v_color, reasons = analyze_verdict(liq, mkt, opt)
col_v1, col_v2 = st.columns([1, 3])
with col_v1:
    if v_color == "green": st.success(f"## {verdict}")
    elif v_color == "red": st.error(f"## {verdict}")
    else: st.info(f"## {verdict}")
with col_v2:
    st.write("**关键因子:** " + ", ".join(reasons))

st.divider()

# 模块 B: 宏观日历 (新增亮点)
st.subheader("📅 关键宏观日程 (Macro Calendar)")

# 将日历横向排列
if cal:
    cols = st.columns(len(cal[:4])) # 只显示最近4个，避免太挤
    for idx, event in enumerate(cal[:4]):
        with cols[idx]:
            days = event['Days']
            label = "今天!" if days == 0 else f"还有 {days} 天"
            color = "inverse" if days <= 3 else "normal"
            st.metric(event['Event'], event['Date'], label, delta_color=color)
else:
    st.write("近期无一级宏观数据发布。")

st.divider()

# 模块 C: 市场数据
st.subheader("📊 市场全景 (Market Overview)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("SOFR 资金成本", f"{liq['SOFR']}%", liq['Status'], delta_color="off")
c2.metric("VIX 恐慌指数", f"{mkt['VIX']:.2f}", f"{mkt['VIX_Chg']:.2f}", delta_color="inverse")
c3.metric("QQQ 期权 PCR", f"{opt['PCR']}", opt['Sentiment'], delta_color="inverse")
c4.metric("美股情绪", f"{mkt['Stock_FNG']}", mkt['Stock_Text'])

st.divider()

# 模块 D: 期权与新闻
col_d1, col_d2 = st.columns([1, 1])

with col_d1:
    st.subheader("⚡ QQQ 期权异动雷达")
    st.caption("筛选: Vol > 1000 且 > 1.5倍持仓 (Smart Money)")
    if opt['Unusual']:
        st.dataframe(pd.DataFrame(opt['Unusual']), hide_index=True, use_container_width=True)
    else:
        st.info("今日无显著机构异动。")

with col_d2:
    st.subheader("📰 智能新闻流")
    if not news_df.empty:
        for i, row in news_df.head(6).iterrows():
            try:
                res = sentiment_model(row['Title'][:512])[0]
                icon = "🟢" if res['label']=='positive' else "🔴" if res['label']=='negative' else "⚪"
                st.markdown(f"{icon} [{row['Title']}]({row['Link']})")
            except: pass
    else:
        st.write("暂无新闻。")
