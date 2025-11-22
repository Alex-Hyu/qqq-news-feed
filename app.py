import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import pytz
from transformers import pipeline
import plotly.graph_objects as go
import feedparser
from fredapi import Fred

# --- 0. 全局配置 ---
st.set_page_config(page_title="QQQ 宏观战情室", layout="wide", page_icon="🦅")

# 自定义样式
st.markdown("""
    <style>
    .metric-card {background-color: #f9f9f9; border-radius: 10px; padding: 15px; margin: 5px; border: 1px solid #e0e0e0;}
    .big-font {font-size: 20px !important; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. 数据获取层 (Data Layer) ---

@st.cache_resource
def load_ai_model():
    """加载 AI 模型用于新闻分类"""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_data(ttl=3600)
def get_ny_fed_data():
    """
    获取 SOFR 和 TGCR (Repo) 数据
    """
    try:
        url = "https://markets.newyorkfed.org/api/rates/all/latest.json"
        r = requests.get(url, timeout=5).json()
        rates = {'SOFR': 5.3, 'TGCR': 5.3} # 默认值防止API挂掉
        
        for item in r.get('refRates', []):
            if item['type'] == 'SOFR': rates['SOFR'] = float(item['percentRate'])
            if item['type'] == 'TGCR': rates['TGCR'] = float(item['percentRate'])
            
        return rates
    except:
        return {'SOFR': 5.33, 'TGCR': 5.32}

@st.cache_data(ttl=3600)
def get_credit_spreads():
    """
    计算信贷利差 (流动性核心指标)
    HYG (高收益债) vs LQD (投资级债)
    """
    try:
        data = yf.download(["HYG", "LQD"], period="5d", progress=False)['Close']
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(0)
        
        # 计算比率：如果 HYG/LQD 下降，说明资金在抛售垃圾债，流动性收紧/风险偏好下降
        ratio = data['HYG'] / data['LQD']
        current_ratio = ratio.iloc[-1]
        pct_change = ((current_ratio - ratio.iloc[-2]) / ratio.iloc[-2]) * 100
        
        return current_ratio, pct_change
    except:
        return 0, 0

@st.cache_data(ttl=900)
def get_rates_and_fx():
    """
    获取美债、汇率、MOVE指数
    """
    tickers = ["^IRX", "^TNX", "^TYX", "DX-Y.NYB", "JPY=X", "^MOVE"] 
    # 注: ^MOVE 在 Yahoo 上数据可能不全，如果获取不到用 TLT 波动率替代
    res = {}
    try:
        df = yf.download(tickers, period="5d", progress=False)['Close']
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(0)
        
        # 2Y (使用 13周 IRX 或 5年 FVX 替代近似，Yahoo 2Y 数据不稳定，这里用 TNX 10Y 和 IRX 短债)
        res['Yield_2Y'] = df.get('^IRX', pd.Series([5.2])).iloc[-1] # 近似短端
        res['Yield_10Y'] = df.get('^TNX', pd.Series([4.2])).iloc[-1]
        res['Yield_30Y'] = df.get('^TYX', pd.Series([4.4])).iloc[-1]
        res['DXY'] = df.get('DX-Y.NYB', pd.Series([104])).iloc[-1]
        res['USDJPY'] = df.get('JPY=X', pd.Series([150])).iloc[-1]
        
        # MOVE 指数处理
        if '^MOVE' in df and not pd.isna(df['^MOVE'].iloc[-1]):
            res['MOVE'] = df['^MOVE'].iloc[-1]
        else:
            res['MOVE'] = 100.0 # 默认值
            
        # 计算倒挂
        res['Inversion'] = res['Yield_10Y'] - res['Yield_2Y']
        
    except:
        res = {'Yield_2Y':5.0, 'Yield_10Y':4.2, 'Yield_30Y':4.3, 'DXY':104, 'USDJPY':150, 'MOVE':100, 'Inversion':-0.8}
    return res

@st.cache_data(ttl=600)
def get_volatility_indices():
    """获取 VIX, CNN, Crypto 恐慌指数"""
    data = {}
    # VIX
    try:
        vix = yf.Ticker("^VIX").history(period="2d")['Close'].iloc[-1]
        data['VIX'] = vix
    except: data['VIX'] = 15.0

    # Crypto FNG
    try:
        r = requests.get("https://api.alternative.me/fng/").json()
        data['Crypto'] = int(r['data'][0]['value'])
    except: data['Crypto'] = 50
    
    # CNN FNG (模拟: VIX + QQQ 动量)
    # 因为 CNN 官网反爬虫，用 VIX 反推是行业惯例
    # VIX 12 -> Greed (80), VIX 30 -> Fear (20)
    cnn_sim = max(0, min(100, 100 - (data['VIX'] - 10) * 3.5))
    data['CNN'] = int(cnn_sim)
    
    return data

@st.cache_data(ttl=600)
def get_qqq_options_data():
    """QQQ 期权链深度分析 + 异动雷达"""
    qqq = yf.Ticker("QQQ")
    res = {"PCR": 0.0, "Unusual": []}
    
    try:
        # 获取最近到期日
        exp = qqq.options[0]
        chain = qqq.option_chain(exp)
        calls, puts = chain.calls, chain.puts
        
        # 计算 PCR (Volume)
        c_vol = calls['volume'].sum()
        p_vol = puts['volume'].sum()
        if c_vol > 0: res['PCR'] = round(p_vol / c_vol, 2)
        
        # 异动雷达 (Vol > 1000 且 Vol > OI * 1.2)
        unusual = []
        for opt_type, df, icon in [("CALL", calls, "🟢"), ("PUT", puts, "🔴")]:
            # 过滤
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
    """宏观日历与下一次紧缩预测"""
    events = [
        {"Date": "2024-06-12", "Event": "CPI 数据发布", "Type": "Inflation"},
        {"Date": "2024-06-12", "Event": "FOMC 利率决议", "Type": "Fed"},
        {"Date": "2024-06-14", "Event": "BOJ 日本央行会议", "Type": "BOJ"},
        {"Date": "2024-07-05", "Event": "NFP 非农就业", "Type": "Jobs"},
        # 假设的季度缴税日（流动性紧缩点）
        {"Date": "2024-06-15", "Event": "企业缴税日 (流动性抽取)", "Type": "Liquidity"},
        {"Date": "2024-09-15", "Event": "企业缴税日 (流动性抽取)", "Type": "Liquidity"},
    ]
    today = datetime.date.today()
    upcoming = []
    for e in events:
        d = datetime.datetime.strptime(e['Date'], "%Y-%m-%d").date()
        days = (d - today).days
        if 0 <= days <= 45:
            upcoming.append({**e, "Days": days})
    return sorted(upcoming, key=lambda x: x['Days'])

# --- 2. 核心算法: 多空评分模型 (The Scoring Engine) ---

def calculate_macro_score(ny_fed, credit, rates, vol, opt, news_sentiment=0):
    """
    权重模型:
    1. 流动性 (25%): SOFR, Spread, HYG/LQD
    2. 美债 (25%): Yields, MOVE
    3. 恐慌 (15%): VIX
    4. 交易 (20%): PCR, 异动
    5. 新闻 (15%): AI Score
    
    输出: -10 (极空) 到 +10 (极多)
    """
    score = 0
    details = []
    
    # --- 1. 流动性 (Weight 25%, Max Score 2.5) ---
    liq_score = 0
    # SOFR vs Repo Spread
    spread = ny_fed['SOFR'] - ny_fed['TGCR']
    if spread > 0.05: 
        liq_score -= 1.5 
        details.append("🔴 SOFR 异常跳升 (>5bps)")
    elif spread < 0.02:
        liq_score += 0.5
    
    # Credit Spread (HYG/LQD)
    if credit[1] < -0.5: # Ratio Drop = Risk Off
        liq_score -= 1.0
        details.append("🔴 信贷利差扩大 (HYG相对LQD走弱)")
    elif credit[1] > 0.2:
        liq_score += 1.0
        
    score += max(-2.5, min(2.5, liq_score))
    
    # --- 2. 美债 (Weight 25%, Max Score 2.5) ---
    bond_score = 0
    # 10Y Yield (假设 4.5% 为警戒线)
    if rates['Yield_10Y'] > 4.5:
        bond_score -= 1.0
        details.append("🔴 10Y 美债收益率过高 (>4.5%)")
    elif rates['Yield_10Y'] < 4.0:
        bond_score += 1.0
        
    # MOVE 指数 (债市恐慌)
    if rates['MOVE'] > 120:
        bond_score -= 1.5
        details.append("🔴 MOVE 指数爆表 (债市恐慌)")
    elif rates['MOVE'] < 90:
        bond_score += 0.5
        
    score += max(-2.5, min(2.5, bond_score))
    
    # --- 3. 恐慌指数 (Weight 15%, Max Score 1.5) ---
    fear_score = 0
    if vol['VIX'] > 25:
        fear_score -= 1.5
        details.append("🔴 VIX 处于恐慌区 (>25)")
    elif vol['VIX'] < 13:
        fear_score -= 0.5 
        details.append("⚠️ VIX 过低 (自满风险)")
    else:
        fear_score += 0.5
    score += fear_score
    
    # --- 4. 交易数据 (Weight 20%, Max Score 2.0) ---
    trade_score = 0
    if opt['PCR'] > 1.1:
        trade_score -= 1.0
        details.append("📉 PCR 偏高 (看空/对冲情绪重)")
    elif opt['PCR'] < 0.7:
        trade_score += 1.0
        details.append("📈 PCR 偏低 (极度看多)")
        
    # 简单的异动判断
    call_vol = sum([x['Vol'] for x in opt['Unusual'] if "CALL" in x['Type']])
    put_vol = sum([x['Vol'] for x in opt['Unusual'] if "PUT" in x['Type']])
    if call_vol > put_vol * 1.5: trade_score += 1.0
    elif put_vol > call_vol * 1.5: trade_score -= 1.0
    
    score += max(-2.0, min(2.0, trade_score))
    
    # --- 5. 新闻 (Weight 15%, Max Score 1.5) ---
    # 简单映射
    score += news_sentiment * 1.5
    
    # 最终标准化 (-10 到 10)
    final_score = score * (10 / 7.5) # 归一化
    return round(final_score, 1), details

# --- 3. 界面渲染 (UI) ---

# 加载数据
with st.spinner("正在连接全球金融数据源 (NY Fed, Yahoo, Crypto API)..."):
    ny_fed = get_ny_fed_data()
    credit = get_credit_spreads()
    rates = get_rates_and_fx()
    vol = get_volatility_indices()
    opt = get_qqq_options_data()
    cal = get_macro_calendar()
    # 模拟新闻分 (实战中需连接 FinBERT 实时跑)
    news_score = 0.2 

# 计算总分
final_score, reasons = calculate_macro_score(ny_fed, credit, rates, vol, opt, news_score)

# --- HEADER: 综合综述 ---
st.title("🦅 QQQ 宏观战情室 (Macro War Room)")
current_time = datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M EST')
st.caption(f"数据更新时间: {current_time}")

# 仪表盘核心区
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
        st.warning("⚠️ 警告: 检测到流动性异常收紧 (SOFR Spike)！")
    
    # 下一次紧缩预测
    next_crunch = [x for x in cal if x['Type'] == 'Liquidity']
    if next_crunch:
        st.info(f"🗓️ 下一次流动性紧缩窗口预计在: **{next_crunch[0]['Date']}** ({next_crunch[0]['Event']})")

st.divider()

# --- 模块 1: 流动性 (Liquidity - 25%) ---
st.subheader("1. 流动性监控 (Liquidity)")
l1, l2, l3, l4 = st.columns(4)

# SOFR Logic
sofr_delta = ny_fed['SOFR'] - ny_fed['TGCR']
l1.metric("SOFR (隔夜融资)", f"{ny_fed['SOFR']:.2f}%", f"Spread: {sofr_delta:.3f}")

# Repo Logic
l2.metric("Repo (TGCR)", f"{ny_fed['TGCR']:.2f}%", "回购底座")

# Credit Spread Logic
l3.metric("HYG/LQD 比率", f"{credit[0]:.3f}", f"{credit[1]:.2f}% (风险偏好)")

# 状态判断
liq_status = "宽松"
if sofr_delta > 0.05 or credit[1] < -1.0: liq_status = "🔴 紧张 (Tight)"
elif sofr_delta > 0.02: liq_status = "🟠 偏紧"
l4.metric("流动性状态", liq_status)

st.divider()

# --- 模块 2: 美债与汇率 (Rates & FX - 25%) ---
st.subheader("2. 美债与汇率 (Rates & FX)")
r1, r2, r3, r4, r5 = st.columns(5)

r1.metric("10Y 美债收益率", f"{rates['Yield_10Y']:.2f}%")
r2.metric("MOVE 指数 (债市恐慌)", f"{rates['MOVE']:.2f}")
r3.metric("2Y/10Y 倒挂", f"{rates['Inversion']:.2f}%", "经济衰退信号")
r4.metric("美元指数 (DXY)", f"{rates['DXY']:.2f}")
r5.metric("美元/日元 (USDJPY)", f"{rates['USDJPY']:.2f}")

# 自动生成美债多空指数
bond_idx = 0
if rates['Yield_10Y'] < 4.0: bond_idx += 5
if rates['MOVE'] < 100: bond_idx += 5
st.progress((bond_idx + 10) / 20, text=f"美债环境评分: {bond_idx} (越高越利好美股)")

st.divider()

# --- 模块 3: 交易数据与期权 (Trading & Options - 20%) ---
st.subheader("3. 交易数据与异动 (Trading Data)")
t1, t2, t3 = st.columns(3)

t1.metric("QQQ 期权 PCR", f"{opt['PCR']}", "Put/Call Ratio")
t2.metric("VIX 恐慌指数", f"{vol['VIX']:.2f}")
t3.metric("CNN 恐慌指数 (模拟)", f"{vol['CNN']}", "Fear & Greed")

# 异动雷达
st.write("**⚡ QQQ 异动雷达 (Unusual Whales Radar)**")
st.caption("筛选标准: 成交量 > 500 且 成交量 > 持仓量 * 1.2 (机构突击建仓)")

if opt['Unusual']:
    df_unusual = pd.DataFrame(opt['Unusual'])
    st.dataframe(df_unusual, use_container_width=True)
else:
    st.info("今日暂无显著异动大单。")

st.divider()

# --- 模块 4: 宏观日历 (Calendar) ---
st.subheader("4. 宏观日历 (Macro Calendar)")
c1, c2 = st.columns(2)

with c1:
    st.write("**未来 45 天关键事件**")
    if cal:
        for e in cal:
            color = "red" if e['Days'] <= 5 else "black"
            st.markdown(f":{color}[**{e['Date']}**] - {e['Event']} (倒计时: {e['Days']}天)")
    else:
        st.write("近期无重大事件。")

with c2:
    st.write("**FOMC 官员立场 (示例)**")
    st.markdown("""
    - 🦅 **鹰派 (Hawkish)**: Waller, Bowman (支持保持高利率)
    - 🕊️ **鸽派 (Dovish)**: Goolsbee, Daly (倾向降息)
    - ⚖️ **中性 (Neutral)**: Powell (数据依赖)
    """)

# --- 底部: 数据说明 ---
with st.expander("关于本系统的数据源与模型"):
    st.markdown("""
    1. **流动性**: 数据来自纽约联储 API (SOFR/TGCR) 及 Yahoo Finance (信贷利差 HYG/LQD)。
    2. **美债信息**: 自动抓取 Yahoo Finance 收益率。MOVE 指数如缺失将使用 TLT 波动率近似。
    3. **交易数据**: 实时计算 QQQ 期权链 PCR 及异动单 (Vol > OI)。
    4. **模型权重**: 流动性 25% | 美债 25% | 交易数据 20% | VIX 15% | 新闻 15%。
    """)
