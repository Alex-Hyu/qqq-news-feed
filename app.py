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

# --- [侧边栏] 配置 ---
with st.sidebar:
    st.header("⚙️ 设置")
    av_api_key = st.text_input("AlphaVantage API Key", value="UMWB63OXOOCIZHXR", type="password")
    st.divider()
    st.subheader("系统状态")
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

# --- [修复] Gamma Flip / Wall 聚合算法 ---
@st.cache_data(ttl=1800)
def get_derivatives_structure():
    res = {
        "Futures_Basis": 0, "Basis_Status": "Normal", 
        "GEX_Net": "Neutral", "Call_Wall": 0, "Put_Wall": 0, 
        "Flip_Line": 0, "Current_Price": 0,
        "Vanna_Charm_Proxy": "Neutral",
        "Data_Note": ""
    }
    try:
        # 1. 基础价格
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
        
        # 2. [核心修复] 聚合多期权链计算 Wall
        qqq = yf.Ticker("QQQ")
        # 获取最近的 4 个到期日 (覆盖周权和月权)
        expirations = qqq.options[:4] 
        
        all_calls = []
        all_puts = []
        
        for date in expirations:
            try:
                chain = qqq.option_chain(date)
                # 必须清洗数据: 填充 NaN 为 0
                c = chain.calls.fillna(0)
                p = chain.puts.fillna(0)
                all_calls.append(c[['strike', 'openInterest', 'volume']])
                all_puts.append(p[['strike', 'openInterest', 'volume']])
            except:
                continue
        
        if all_calls and all_puts:
            # 合并数据
            df_calls = pd.concat(all_calls)
            df_puts = pd.concat(all_puts)
            
            # 按 Strike 聚合求和 OI
            total_calls = df_calls.groupby('strike')['openInterest'].sum()
            total_puts = df_puts.groupby('strike')['openInterest'].sum()
            
            # 找到聚合后的最大持仓位
            res['Call_Wall'] = total_calls.idxmax()
            res['Put_Wall'] = total_puts.idxmax()
            
            # 3. 计算 Flip Line
            # 算法: Call OI - Put OI 的差值 (Net Gamma Proxy)
            # 对齐索引
            combined = pd.DataFrame({'Call_OI': total_calls, 'Put_OI': total_puts}).fillna(0)
            combined['Net_OI'] = combined['Call_OI'] - combined['Put_OI']
            
            # 寻找符号翻转点 (从正变负的地方)
            # 或者寻找 Net OI 最接近 0 的点 (在 Put Wall 和 Call Wall 之间)
            # 简单算法: 寻找 Put OI 开始超过 Call OI 的关键点
            flip_candidates = combined[combined['Net_OI'] < 0]
            if not flip_candidates.empty:
                # 找最接近现价的翻转点
                flip_strike = flip_candidates.index[0] # 简易取第一个
                # 优化: 在现价附近找
                near_price = flip_candidates.index[abs(flip_candidates.index - qqq_price).argmin()]
                res['Flip_Line'] = near_price
            else:
                res['Flip_Line'] = res['Put_Wall'] # 兜底
                
            # GEX 状态判定
            if qqq_price < res['Flip_Line']: res['GEX_Net'] = "🔴 Negative (高波)"
            else: res['GEX_Net'] = "🟢 Positive (低波)"
            
            res['Data_Note'] = f"聚合了 {len(expirations)} 个到期日"
            
        # Vanna
        if market_data['^NDX'].iloc[-1] > market_data['^NDX'].iloc[-2]:
            res['Vanna_Charm_Proxy'] = "Tailwind (助涨)"
        else: res['Vanna_Charm_Proxy'] = "Headwind (阻力)"

    except Exception as e: 
        res['Data_Note'] = "数据获取失败"
        print(e)
    return res

# --- [修复] PCR 计算也改为聚合模式 ---
@st.cache_data(ttl=1800)
def get_qqq_options_data():
    qqq = yf.Ticker("QQQ")
    res = {"PCR": 0.0, "Unusual": []}
    try:
        # 同样聚合前 4 个到期日，样本量更大更准
        expirations = qqq.options[:4]
        
        total_c_vol = 0
        total_p_vol = 0
        unusual = []
        
        for date in expirations:
            try:
                chain = qqq.option_chain(date)
                calls = chain.calls.fillna(0)
                puts = chain.puts.fillna(0)
                
                total_c_vol += calls['volume'].sum()
                total_p_vol += puts['volume'].sum()
                
                # 异动扫描 (只保留真正的大单)
                for opt_type, df, icon in [("CALL", calls, "🟢"), ("PUT", puts, "🔴")]:
                    # 提高阈值: 成交量 > 1000
                    hot = df[(df['volume'] > 1000) & (df['volume'] > df['openInterest'] * 1.5)]
                    for _, row in hot.iterrows():
                        unusual.append({
                            "Type": f"{icon} {opt_type}", 
                            "Strike": row['strike'],
                            "Exp": date, # 加上日期
                            "Vol": int(row['volume']), 
                            "OI": int(row['openInterest']),
                            "Ratio": round(row['volume'] / (row['openInterest']+1), 1)
                        })
            except: continue
            
        if total_c_vol > 0: 
            res['PCR'] = round(total_p_vol / total_c_vol, 2)
            
        # 按成交量排序取前 15
        res['Unusual'] = sorted(unusual, key=lambda x: x['Vol'], reverse=True)[:15]
    except: pass
    return res

# 日历 (Alpha Vantage)
@st.cache_data(ttl=3600)
def get_macro_calendar(api_key=""):
    if api_key:
        try:
            url = f"https://www.alphavantage.co/query?function=ECONOMIC_CALENDAR&apikey={api_key}"
            r = requests.get(url, timeout=5)
            df = pd.read_csv(StringIO(r.text))
            df = df[df['currency'] == 'USD']
            keywords = ["GDP", "Unemployment", "CPI", "Interest Rate", "Payroll", "FOMC", "PCE"]
            df['is_important'] = df['event'].apply(lambda x: any(k in x for k in keywords))
            df = df[df['is_important']]
            today = datetime.date.today().strftime("%Y-%m-%d")
            df = df[df['date'] >= today].sort_values('date').head(10)
            display_df = df[['date', 'time', 'event', 'estimate', 'previous']].copy()
            display_df.columns = ['Date', 'Time', 'Event', 'Est', 'Prev']
            if not display_df.empty: return display_df, "API Data"
        except: pass

    # 备用
    today = datetime.date.today()
    events = []
    next_month = today.replace(day=28) + datetime.timedelta(days=4)
    next_cpi = today.replace(day=12) 
    if today.day > 12: next_cpi = (next_month - datetime.timedelta(days=1)).replace(day=12)
    events.append({"Date": next_cpi, "Event": "CPI (Est)", "Type": "Inflation"})
    
    events = sorted(events, key=lambda x: x['Date'])
    df = pd.DataFrame(events)
    df = df[df['Date'] >= today].head(5)
    d_df = df.copy()
    d_df['Time']="--"; d_df['Est']="--"; d_df['Prev']="--"
    return d_df[['Date','Time','Event','Est','Prev']], "Estimated"

@st.cache_data(ttl=1800)
def get_macro_news():
    feeds = [
        ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("MarketWatch", "http://feeds.marketwatch.com/marketwatch/topstories/")
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
    
    # 流动性 (25%)
    liq_score = 0
    spread = ny_fed['SOFR'] - ny_fed['TGCR']
    if spread > 0.05: liq_score -= 1.0; details.append("🔴 SOFR 异常")
    elif spread < 0.02: liq_score += 0.5
    if fed_liq['RRP_Chg'] > 20: liq_score -= 0.5; details.append("🔴 RRP 抽水")
    if fed_liq['TGA_Chg'] > 20: liq_score -= 0.5; details.append("🔴 TGA 抽水")
    if credit[1] < -0.5: liq_score -= 0.5
    score += max(-2.5, min(2.5, liq_score))
    
    # 美债 (25%)
    bond_score = 0
    if rates['Yield_10Y'] > 4.5: bond_score -= 1.0
    elif rates['Yield_10Y'] < 4.0: bond_score += 1.0
    if rates['MOVE'] > 110: bond_score -= 1.5
    score += max(-2.5, min(2.5, bond_score))
    
    # 恐慌 (15%)
    fear_score = 0
    if vol['VIX'] > 25: fear_score -= 1.0
    elif vol['VIX'] < 13: fear_score -= 0.5
    if vol['Crypto_Val'] < 20: fear_score += 0.5
    score += fear_score
    
    # 交易 (20%)
    trade_score = 0
    if opt['PCR'] > 1.1: trade_score -= 0.5; details.append("📉 PCR 偏空")
    elif opt['PCR'] < 0.7: trade_score += 0.5
    if deriv['Basis_Status'].startswith("🔴"): trade_score -= 1.0; details.append("🔴 期货贴水")
    if deriv['GEX_Net'].startswith("🔴"): trade_score -= 0.5; details.append("🔴 跌破 Gamma Flip")
    elif deriv['GEX_Net'].startswith("🟢"): trade_score += 0.5
    score += max(-2.0, min(2.0, trade_score))
    
    # 新闻 (15%)
    score += news_score_val * 1.5
    
    return round(score * (10 / 7.5), 1), details

# --- 3. UI ---

with st.spinner("正在聚合多期权链数据 (30分钟刷新)..."):
    ai_model = load_ai_model()
    ny_fed = get_ny_fed_data()
    fed_liq = get_fed_liquidity()
    credit = get_credit_spreads()
    rates = get_rates_and_fx()
    vol = get_volatility_indices()
    # 核心数据源
    opt = get_qqq_options_data()
    deriv = get_derivatives_structure()
    
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

# 3. 交易与微观结构 (聚合版)
st.subheader("3. 交易与微观结构 (Aggregated Options & GEX)")
st.caption(f"数据说明: 已聚合未来 4 个到期日 (包含月权) 的 OI 数据，解决 0DTE 数据缺失问题。")

t1, t2, t3, t4 = st.columns(4)
t1.metric("QQQ 期权 PCR", f"{opt['PCR']}", "Put/Call Ratio")
t2.metric("VIX 股市恐慌", f"{vol['VIX']:.2f}")
t3.metric("币圈恐慌指数", f"{vol['Crypto_Val']}", f"{vol['Crypto_Text']}")
t4.metric("期货基差 (Basis)", f"{deriv['Futures_Basis']:.2f}", deriv['Basis_Status'])

g1, g2, g3 = st.columns(3)
g1.metric("Gamma Flip Line (聚合自算)", f"${deriv['Flip_Line']:.2f}", deriv['GEX_Net'], delta_color="off")
g2.metric("Put Wall (强支撑)", f"${deriv['Put_Wall']}", "Total OI Max")
g3.metric("Call Wall (强阻力)", f"${deriv['Call_Wall']}", "Total OI Max")

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
        st.write("**⚡ 异动雷达 (Aggregated Volume > 1000)**")
        if opt['Unusual']: 
            st.dataframe(
                pd.DataFrame(opt['Unusual']), 
                column_config={"Exp": "到期日"},
                use_container_width=True
            )
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
        st.dataframe(cal_df, hide_index=True, use_container_width=True)
    else: st.write("近期无重要数据。")

with c2:
    st.markdown("""
    **Fed 观察**:
    - 🦅 **鹰派**: Waller
    - 🕊️ **鸽派**: Goolsbee
    - ⚖️ **中性**: Powell
    """)
