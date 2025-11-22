import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline
import datetime
import feedparser
import requests
import numpy as np

# --- 页面配置 ---
st.set_page_config(page_title="QQQ 机构宏观雷达", layout="wide", page_icon="🦅")

# --- 缓存加载 ---
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# --- 1. 宏观流动性数据 (SOFR/Repo) ---
@st.cache_data(ttl=3600)
def get_liquidity_data():
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
        return rates
    except:
        return {'SOFR': 5.30, 'TGCR': 5.30}

# --- 2. 恐慌指数与 VIX ---
@st.cache_data(ttl=600) # 10分钟更新
def get_market_emotion():
    data = {}
    # 币圈
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=5)
        d = r.json()
        data['Crypto'] = int(d['data'][0]['value'])
    except: data['Crypto'] = 50
    
    # VIX 与 股价
    try:
        tickers = yf.download(["^VIX", "QQQ"], period="2d", progress=False)['Close']
        if isinstance(tickers.columns, pd.MultiIndex): tickers.columns = tickers.columns.droplevel(0)
        
        data['VIX'] = tickers['^VIX'].iloc[-1]
        
        # 计算 VIX 变动
        vix_prev = tickers['^VIX'].iloc[-2]
        data['VIX_Change'] = round(data['VIX'] - vix_prev, 2)
        
        # QQQ 涨跌
        qqq_curr = tickers['QQQ'].iloc[-1]
        qqq_prev = tickers['QQQ'].iloc[-2]
        data['QQQ_Change'] = ((qqq_curr - qqq_prev) / qqq_prev) * 100
        
    except:
        data['VIX'] = 0
        data['VIX_Change'] = 0
        data['QQQ_Change'] = 0
        
    return data

# --- 3. QQQ 期权链深度分析 (核心升级) ---
@st.cache_data(ttl=600)
def get_qqq_options_analysis():
    """
    获取 QQQ 最近两个到期日的期权链，计算 PCR 和 异动
    """
    qqq = yf.Ticker("QQQ")
    analysis = {"PCR_Volume": 0, "PCR_OI": 0, "Unusual": []}
    
    try:
        # 获取最近的两个到期日 (例如本周五和下周五)
        expirations = qqq.options[:2]
        
        total_call_vol = 0
        total_put_vol = 0
        total_call_oi = 0
        total_put_oi = 0
        
        unusual_trades = []
        
        for date in expirations:
            chain = qqq.option_chain(date)
            calls = chain.calls
            puts = chain.puts
            
            # 1. 累加数据计算 PCR (Put/Call Ratio)
            total_call_vol += calls['volume'].sum()
            total_put_vol += puts['volume'].sum()
            total_call_oi += calls['openInterest'].sum()
            total_put_oi += puts['openInterest'].sum()
            
            # 2. 扫描异动 (筛选标准: 成交量 > 500 且 成交量 > 未平仓数 * 1.2)
            # 逻辑：如果今天的成交量比所有的持仓量还大，说明有巨大的新资金进场
            
            # 扫描 Call
            active_calls = calls[(calls['volume'] > 500) & (calls['volume'] > calls['openInterest'] * 1.2)]
            for _, row in active_calls.iterrows():
                unusual_trades.append({
                    "Type": "CALL 🟢",
                    "Strike": row['strike'],
                    "Exp": date,
                    "Vol": int(row['volume']),
                    "OI": int(row['openInterest']),
                    "Vol/OI": round(row['volume'] / (row['openInterest']+1), 1)
                })
                
            # 扫描 Put
            active_puts = puts[(puts['volume'] > 500) & (puts['volume'] > puts['openInterest'] * 1.2)]
            for _, row in active_puts.iterrows():
                unusual_trades.append({
                    "Type": "PUT 🔴",
                    "Strike": row['strike'],
                    "Exp": date,
                    "Vol": int(row['volume']),
                    "OI": int(row['openInterest']),
                    "Vol/OI": round(row['volume'] / (row['openInterest']+1), 1)
                })
        
        # 计算比率
        if total_call_vol > 0: analysis['PCR_Volume'] = round(total_put_vol / total_call_vol, 2)
        if total_call_oi > 0: analysis['PCR_OI'] = round(total_put_oi / total_call_oi, 2)
        
        # 按成交量排序异动
        analysis['Unusual'] = sorted(unusual_trades, key=lambda x: x['Vol'], reverse=True)[:10]
        
        return analysis
        
    except Exception as e:
        print(e)
        return analysis

# --- 4. 新闻获取 ---
@st.cache_data(ttl=300)
def get_news_headlines():
    feeds = [
        ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("MarketWatch", "http://feeds.marketwatch.com/marketwatch/topstories/")
    ]
    all_news = []
    for name, url in feeds:
        try:
            f = feedparser.parse(url)
            for e in f.entries[:3]:
                all_news.append({"Title": e.title, "Link": e.link, "Source": name})
        except: pass
    return pd.DataFrame(all_news)

# --- 5. 综合研判算法 ---
def get_final_verdict(sofr, vix, pcr_vol, news_sentiment):
    score = 0
    reasons = []
    
    # 1. 流动性 (权重 30%)
    if sofr > 5.4: 
        score -= 2
        reasons.append("流动性紧张 (SOFR高)")
    else:
        score += 1
        
    # 2. VIX (权重 20%)
    if vix > 25: 
        score -= 2
        reasons.append("市场极度恐慌 (VIX>25)")
    elif vix < 13:
        score -= 1
        reasons.append("市场过于自满 (VIX<13)")
    else:
        score += 1
        
    # 3. 期权情绪 (PCR) (权重 30%)
    # PCR > 1.0 说明 Put 多，市场看空 (但也可能是底部)
    # PCR < 0.6 说明 Call 多，市场极度看多
    if pcr_vol > 1.1:
        score -= 1
        reasons.append("期权交易者偏空 (PCR > 1.1)")
    elif pcr_vol < 0.6:
        score += 1
        reasons.append("期权交易者偏多 (PCR < 0.6)")
        
    # 4. 新闻情绪 (权重 20%)
    if news_sentiment == "Bullish": score += 2
    if news_sentiment == "Bearish": score -= 2
    
    # 结论
    if score >= 3: return "强力做多 (Strong Bull)", "green", reasons
    elif score >= 1: return "谨慎看多 (Bullish)", "lightgreen", reasons
    elif score <= -3: return "强力做空 (Strong Bear)", "red", reasons
    elif score <= -1: return "谨慎看空 (Bearish)", "lightcoral", reasons
    else: return "中性震荡 (Neutral)", "gray", reasons

# --- 主界面渲染 ---
st.title("🦅 QQQ 全维宏观对冲终端")
st.caption("数据源: NY Fed (流动性) | Yahoo Finance (期权/价格) | RSS (新闻)")

with st.spinner("正在加载全市场数据 (期权链计算较慢，请稍候)..."):
    liq = get_liquidity_data()
    emo = get_market_emotion()
    opt = get_qqq_options_analysis()
    news = get_news_headlines()
    sentiment_pipe = load_sentiment_model()

# --- 区域 1: 核心仪表盘 ---
st.subheader("📊 核心指标 (Key Metrics)")
c1, c2, c3, c4 = st.columns(4)

# VIX
vix_color = "inverse" # VIX涨是坏事
c1.metric("VIX 恐慌指数", f"{emo['VIX']:.2f}", f"{emo['VIX_Change']}", delta_color=vix_color)

# SOFR
c2.metric("SOFR 资金成本", f"{liq['SOFR']:.2f}%", "流动性基准", delta_color="off")

# PCR (Put/Call Ratio)
pcr_val = opt.get('PCR_Volume', 0)
pcr_delta = "偏空" if pcr_val > 1 else "偏多"
c3.metric("期权多空比 (PCR)", f"{pcr_val}", pcr_delta)

# QQQ 价格
c4.metric("QQQ 现价变动", f"{emo['QQQ_Change']:.2f}%")

st.divider()

# --- 区域 2: 期权深度分析 ---
st.subheader("⚡ QQQ 期权异动 (Smart Money Flow)")

col_opt1, col_opt2 = st.columns([1, 2])

with col_opt1:
    st.info("💡 **数据说明**: 此列表筛选出 **今日成交量 > 持仓量** 的合约。这通常代表机构突击建仓的新资金。")
    st.markdown(f"**总 Put/Call 持仓比 (PCR OI):** `{opt.get('PCR_OI', 0)}`")
    if opt.get('PCR_OI', 0) > 1.5:
        st.warning("⚠️ 市场累积了大量看跌期权 (Heavy Hedging)")
    elif opt.get('PCR_OI', 0) < 0.7:
        st.success("🚀 市场持仓极度看涨 (Bullish Positioning)")

with col_opt2:
    if opt['Unusual']:
        df_unusual = pd.DataFrame(opt['Unusual'])
        st.dataframe(
            df_unusual, 
            column_config={
                "Type": "方向",
                "Strike": "行权价",
                "Exp": "到期日",
                "Vol": "今日成交",
                "OI": "未平仓",
                "Vol/OI": "爆发系数 (Vol/OI)"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.write("今日暂无显著异常大单。")

st.divider()

# --- 区域 3: 新闻与最终研判 ---
st.subheader("🧠 宏观 + 舆情综合研判")

# 简单的 AI 情绪计算
bull_sents = 0
bear_sents = 0
if not news.empty:
    for t in news['Title']:
        try:
            res = sentiment_pipe(t[:512])[0]
            if res['label'] == 'positive': bull_sents += 1
            if res['label'] == 'negative': bear_sents += 1
        except: pass

news_verdict = "Neutral"
if bull_sents > bear_sents: news_verdict = "Bullish"
elif bear_sents > bull_sents: news_verdict = "Bearish"

# 调用最终算法
verdict, v_color, reasons = get_final_verdict(
    liq['SOFR'], emo['VIX'], opt.get('PCR_Volume', 1), news_verdict
)

c_res1, c_res2 = st.columns([1, 1])

with c_res1:
    if v_color == "green": st.success(f"## {verdict}")
    elif v_color == "red": st.error(f"## {verdict}")
    else: st.info(f"## {verdict}")
    
    st.markdown("#### 决策因子:")
    for r in reasons:
        st.write(f"- {r}")

with c_res2:
    st.write("#### 最新关键新闻")
    for i, row in news.iterrows():
        st.markdown(f"• [{row['Title']}]({row['Link']})")
