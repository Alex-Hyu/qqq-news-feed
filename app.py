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

# 自定义 CSS 让界面更紧凑
st.markdown("""
    <style>
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. 基础数据加载与缓存 ---

@st.cache_resource
def load_sentiment_model():
    """加载 FinBERT 模型 (只加载一次)"""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_data(ttl=3600)
def get_macro_liquidity():
    """
    功能1: 获取 SOFR 和 TGCR (Repo) 数据，判断流动性
    来源: 纽约联储 API
    """
    try:
        url = "https://markets.newyorkfed.org/api/rates/all/latest.json"
        r = requests.get(url, timeout=5)
        data = r.json()
        rates = {}
        for item in data.get('refRates', []):
            if item['type'] == 'SOFR': rates['SOFR'] = float(item['percentRate'])
            if item['type'] == 'TGCR': rates['TGCR'] = float(item['percentRate']) # Tri-Party General Collateral Rate
            
        # 兜底数据
        if 'SOFR' not in rates: rates['SOFR'] = 5.30
        if 'TGCR' not in rates: rates['TGCR'] = 5.30
        
        # 流动性判断逻辑
        # 正常情况下 SOFR 和 TGCR 应该非常接近。
        # 如果 SOFR 比 TGCR 高出很多 (>0.05)，说明借钱变难，流动性紧张。
        spread = rates['SOFR'] - rates['TGCR']
        rates['Spread'] = spread
        
        if spread > 0.10: rates['Status'] = "🔴 紧张 (Stress)"
        elif spread > 0.05: rates['Status'] = "🟠 偏紧 (Tight)"
        elif rates['SOFR'] < 4.0: rates['Status'] = "🟢 极度宽松 (Loose)"
        else: rates['Status'] = "⚪ 平稳 (Neutral)"
            
        return rates
    except:
        return {'SOFR': 5.30, 'TGCR': 5.30, 'Spread': 0, 'Status': "数据不可用"}

@st.cache_data(ttl=600)
def get_market_sentiment_data():
    """
    功能2: 获取 VIX, QQQ价格, 币圈恐慌指数, 估算股市恐慌指数
    """
    res = {}
    
    # A. 币圈恐慌 (API)
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=3)
        d = r.json()
        res['Crypto_FNG'] = int(d['data'][0]['value'])
        res['Crypto_Text'] = d['data'][0]['value_classification']
    except:
        res['Crypto_FNG'] = 50
        res['Crypto_Text'] = "Unknown"

    # B. 股市数据 (VIX & QQQ)
    try:
        df = yf.download(["^VIX", "QQQ"], period="5d", progress=False)['Close']
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(0)
        
        # VIX
        res['VIX'] = df['^VIX'].iloc[-1]
        res['VIX_Chg'] = df['^VIX'].iloc[-1] - df['^VIX'].iloc[-2]
        
        # QQQ
        res['QQQ_Price'] = df['QQQ'].iloc[-1]
        res['QQQ_Pct'] = ((df['QQQ'].iloc[-1] - df['QQQ'].iloc[-2]) / df['QQQ'].iloc[-2]) * 100
        
        # C. 模拟 CNN 恐慌指数 (因为 CNN 封锁了 API)
        # 算法: 基于 VIX (恐慌) 和 动量 (Momentum)
        # VIX 12 = 85分(贪婪), VIX 30 = 15分(恐慌)
        vix_score = max(0, min(100, 100 - (res['VIX'] - 10) * 4))
        
        # 动量: 现价 vs 5日均线
        ma5 = df['QQQ'].mean()
        mom_score = 50
        if res['QQQ_Price'] > ma5: mom_score = 70
        else: mom_score = 30
        
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
    """
    功能3 & 4: QQQ 期权链分析 (PCR + 异动雷达)
    """
    qqq = yf.Ticker("QQQ")
    data = {"PCR": 0, "Sentiment": "Neutral", "Unusual": []}
    
    try:
        # 获取最近的 expiration
        exps = qqq.options[:1] # 只看最近一期，保证速度
        
        call_vol, put_vol = 0, 0
        unusual_list = []
        
        for date in exps:
            chain = qqq.option_chain(date)
            calls = chain.calls
            puts = chain.puts
            
            call_vol += calls['volume'].sum()
            put_vol += puts['volume'].sum()
            
            # 异动扫描逻辑: Vol > 1000 且 Vol > OI * 1.5 (大量新开仓)
            # Calls
            hot_calls = calls[(calls['volume']>1000) & (calls['volume'] > calls['openInterest']*1.5)]
            for _, r in hot_calls.iterrows():
                unusual_list.append({
                    "Type": "CALL 🟢", "Strike": r['strike'], "Exp": date, 
                    "Vol": r['volume'], "OI": r['openInterest'], "Ratio": round(r['volume']/(r['openInterest']+1), 1)
                })
            # Puts
            hot_puts = puts[(puts['volume']>1000) & (puts['volume'] > puts['openInterest']*1.5)]
            for _, r in hot_puts.iterrows():
                unusual_list.append({
                    "Type": "PUT 🔴", "Strike": r['strike'], "Exp": date, 
                    "Vol": r['volume'], "OI": r['openInterest'], "Ratio": round(r['volume']/(r['openInterest']+1), 1)
                })
                
        # 计算 PCR
        if call_vol > 0:
            pcr = put_vol / call_vol
            data['PCR'] = round(pcr, 2)
            if pcr > 1.2: data['Sentiment'] = "看空/对冲 (Bearish)"
            elif pcr < 0.7: data['Sentiment'] = "极度看多 (Bullish)"
            else: data['Sentiment'] = "中性 (Neutral)"
            
        data['Unusual'] = sorted(unusual_list, key=lambda x: x['Vol'], reverse=True)[:10]
        
    except Exception as e:
        pass
    return data

@st.cache_data(ttl=3600)
def get_calendar_events():
    """
    功能6: 重大宏观事件提醒 (手动维护关键日期列表 + 动态计算倒计时)
    """
    # 这里列出 2024-2025 关键已知日期 (示例)
    # 实际应用中可以接入 API，但为了免费稳定，我们用静态表 + 倒计时
    events = [
        {"Event": "FOMC 美联储议息", "Date": "2024-06-12"},
        {"Event": "FOMC 美联储议息", "Date": "2024-07-31"},
        {"Event": "FOMC 美联储议息", "Date": "2024-09-18"}, # 假设
        {"Event": "BOJ 日本央行会议", "Date": "2024-06-14"},
        {"Event": "US CPI 通胀数据", "Date": "2024-06-12"},
    ]
    
    today = datetime.date.today()
    upcoming = []
    
    for e in events:
        e_date = datetime.datetime.strptime(e['Date'], "%Y-%m-%d").date()
        days_left = (e_date - today).days
        if 0 <= days_left <= 30: # 只显示未来30天内的
            upcoming.append({
                "Event": e['Event'],
                "Date": e['Date'],
                "Days": days_left,
                "Urgency": "high" if days_left <= 3 else "low"
            })
            
    return sorted(upcoming, key=lambda x: x['Days'])

@st.cache_data(ttl=300)
def get_news_analysis():
    """
    功能5: 新闻获取与 FinBERT 多空标注
    """
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

# --- 2. 核心研判逻辑 (The Brain) ---
def analyze_verdict(liquidity, market_data, options_data, sentiment_data):
    """
    功能7: 综合多空研判
    """
    score = 0
    reasons = []
    
    # 1. 流动性 (一票否决权)
    if liquidity['Status'].startswith("🔴"):
        score -= 3
        reasons.append("❌ 流动性危机预警 (SOFR异常)")
    elif liquidity['Status'].startswith("🟢"):
        score += 1
        reasons.append("✅ 资金面宽松")
        
    # 2. 市场情绪 (VIX & FNG)
    vix = market_data['VIX']
    if vix > 28:
        score -= 2
        reasons.append("❌ 市场极度恐慌 (VIX爆表)")
    elif vix < 12:
        score -= 1
        reasons.append("⚠️ 市场过于自满 (反向指标)")
    elif market_data['Stock_FNG'] < 20:
        score += 2
        reasons.append("✅ 极度恐慌时的超卖反弹机会")
        
    # 3. 期权结构
    pcr = options_data['PCR']
    if pcr > 1.1:
        score -= 1
        reasons.append("📉 期权市场在对冲下跌 (High PCR)")
    elif pcr < 0.6:
        score += 1
        reasons.append("📈 交易员极度看涨 (Low PCR)")
        
    # 4. 趋势
    trend = market_data['QQQ_Pct']
    if trend < -1.5: reasons.append("📉 今日大盘显著下跌")
    if trend > 1.5: reasons.append("📈 今日大盘强势上涨")
    
    # 结论生成
    final_verdict = "中性震荡 (Neutral)"
    color = "gray"
    
    if score >= 2:
        final_verdict = "偏多 (Bullish)"
        color = "green"
    elif score >= 4:
        final_verdict = "强力做多 (Strong Buy)"
        color = "green"
    elif score <= -2:
        final_verdict = "偏空 (Bearish)"
        color = "red"
    elif score <= -4:
        final_verdict = "强力做空 (Strong Sell)"
        color = "red"
        
    return final_verdict, color, reasons

# --- 3. 界面渲染 (UI) ---

# 加载数据
with st.spinner("正在连接全球市场数据源..."):
    liq = get_macro_liquidity()
    mkt = get_market_sentiment_data()
    opt = get_options_radar()
    cal = get_calendar_events()
    news_df = get_news_analysis()
    sentiment_model = load_sentiment_model()

# 顶部标题区
st.title("🦅 QQQ 宏观战情室 (Macro War Room)")
st.markdown(f"**最后更新:** {datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%H:%M EST')}")

# --- 模块 A: 终极研判 (Verdict) ---
verdict, v_color, reasons = analyze_verdict(liq, mkt, opt, "Neutral")
st.markdown("### 🛡️ 综合态势研判")

col_v1, col_v2 = st.columns([1, 3])
with col_v1:
    if v_color == "green":
        st.success(f"## {verdict}")
    elif v_color == "red":
        st.error(f"## {verdict}")
    else:
        st.info(f"## {verdict}")
with col_v2:
    st.write("**关键决策因子:**")
    st.write(", ".join(reasons))

st.divider()

# --- 模块 B: 宏观硬指标 (Liquidity & Calendar) ---
st.subheader("1. 宏观流动性与日历 (Liquidity & Calendar)")
c1, c2, c3, c4 = st.columns(4)

# SOFR
c1.metric("SOFR (资金成本)", f"{liq['SOFR']}%", f"Spread: {liq['Spread']:.3f}", delta_color="inverse")
# 流动性状态
c2.metric("流动性状态", liq['Status'], "GC Repo Monitor")
# 事件提醒
if cal:
    next_event = cal[0]
    c3.metric("下个重大事件", next_event['Event'], f"还有 {next_event['Days']} 天")
else:
    c3.metric("下个重大事件", "暂无近期关注", "30天内")
# VIX
c4.metric("VIX 恐慌指数", f"{mkt['VIX']:.2f}", f"{mkt['VIX_Chg']:.2f}", delta_color="inverse")

st.divider()

# --- 模块 C: 市场情绪与期权 (Sentiment & Options) ---
st.subheader("2. 情绪与期权异动 (Sentiment & Flow)")
c_s1, c_s2, c_s3, c_s4 = st.columns(4)

c_s1.metric("美股恐慌指数", f"{mkt['Stock_FNG']}", mkt['Stock_Text'])
c_s2.metric("币圈恐慌指数", f"{mkt['Crypto_FNG']}", mkt['Crypto_Text'])
c_s3.metric("QQQ 期权 PCR", f"{opt['PCR']}", opt['Sentiment'], delta_color="inverse")
c_s4.metric("QQQ 现价", f"${mkt['QQQ_Price']:.2f}", f"{mkt['QQQ_Pct']:.2f}%")

# 异动雷达表
st.write("**⚡ QQQ 期权异动雷达 (今日成交量 > 持仓量爆发现象)**")
if opt['Unusual']:
    df_unusual = pd.DataFrame(opt['Unusual'])
    st.dataframe(
        df_unusual,
        column_config={
            "Type": "方向", "Strike": "行权价", "Exp": "到期", 
            "Vol": "今日成交", "OI": "原有持仓", "Ratio": "爆发倍数"
        },
        hide_index=True, use_container_width=True
    )
else:
    st.info("今日市场平静，暂无机构突击建仓痕迹。")

st.divider()

# --- 模块 D: 智能新闻流 (Smart News Feed) ---
st.subheader("3. 宏观新闻多空扫描 (AI Scanned News)")

# 预处理新闻情绪
if not news_df.empty:
    # 进度条体验
    progress_text = "AI 正在逐条阅读新闻..."
    my_bar = st.progress(0, text=progress_text)
    
    processed_news = []
    for i, row in news_df.iterrows():
        try:
            # 限制长度防止报错
            res = sentiment_model(row['Title'][:512])[0]
            label = res['label']
            score = res['score']
            
            # 简单的多空转换
            impact = "⚪ 中性"
            if label == "positive" and score > 0.8: impact = "🟢 利多"
            if label == "negative" and score > 0.8: impact = "🔴 利空"
            
            processed_news.append({**row, "Signal": impact})
        except: pass
        my_bar.progress((i+1)/len(news_df), text=progress_text)
    
    my_bar.empty()
    
    # 显示
    col_news1, col_news2 = st.columns(2)
    
    # 分栏显示利好利空
    df_final = pd.DataFrame(processed_news)
    
    with col_news1:
        st.markdown("#### 🔥 重点关注")
        for i, row in df_final.iterrows():
            st.markdown(f"**{row['Signal']}** | [{row['Title']}]({row['Link']})")
            st.caption(f"{row['Source']}")
            
    with col_news2:
        st.markdown("#### 📅 重大事件日历 (模拟数据)")
        if cal:
            for e in cal:
                color = "red" if e['Days'] <= 3 else "gray"
                st.markdown(f":{color}[**{e['Event']}**] - {e['Date']} (还剩 {e['Days']} 天)")
        else:
            st.write("未来30天无一级宏观事件。")

else:
    st.write("暂无最新新闻。")
