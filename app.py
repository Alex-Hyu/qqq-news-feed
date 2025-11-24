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
    .summary-box {padding: 15px; border-radius: 10px; margin-bottom: 20px;}
    .summary-bull {background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;}
    .summary-bear {background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;}
    .summary-neutral {background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db;}
    </style>
    """, unsafe_allow_html=True)

# --- [侧边栏] ---
with st.sidebar:
    st.header("⚙️ 设置")
    av_api_key = st.text_input("AlphaVantage API Key", value="UMWB63OXOOCIZHXR", type="password")
    st.divider()
    st.subheader("系统状态")
    count = st_autorefresh(interval=30 * 60 * 1000, key="data_refresher")
    st.caption(f"🟢 自动刷新: 开启 (30分钟)")
    if st.button("🔄 立即刷新"):
        st.rerun()

# --- 1. 核心数据获取 ---

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
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        ratio = data['HYG'] / data['LQD']
        curr = ratio.iloc[-1]
        pct = ((curr - ratio.iloc[-2]) / ratio.iloc[-2]) * 100
        return curr, pct
    except Exception as e: return 0, 0

# 美债
@st.cache_data(ttl=1800)
def get_rates_and_fx():
    tickers = ["^IRX", "^TNX", "^TYX", "DX-Y.NYB", "JPY=X", "^MOVE"] 
    res = {'Yield_2Y': 0, 'Yield_10Y': 0, 'Inversion': 0, 'DXY': 0, 'USDJPY': 0, 'MOVE': 0}
    try:
        df = yf.download(tickers, period="5d", progress=False)['Close']
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        if '^TNX' in df.columns: res['Yield_10Y'] = df['^TNX'].iloc[-1]
        if '^IRX' in df.columns: res['Yield_2Y'] = df['^IRX'].iloc[-1]
        if res['Yield_10Y'] and res['Yield_2Y']: res['Inversion'] = res['Yield_10Y'] - res['Yield_2Y']
        if 'DX-Y.NYB' in df.columns: res['DXY'] = df['DX-Y.NYB'].iloc[-1]
        if 'JPY=X' in df.columns: res['USDJPY'] = df['JPY=X'].iloc[-1]
        if '^MOVE' in df.columns and not pd.isna(df['^MOVE'].iloc[-1]): res['MOVE'] = df['^MOVE'].iloc[-1]
        else: res['MOVE'] = 100.0
    except Exception as e: pass
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

# --- [重写] Gamma & Vanna/Charm 逻辑 ---
@st.cache_data(ttl=1800)
def get_derivatives_structure():
    res = {
        "Futures_Basis": 0, "Basis_Status": "Normal", 
        "GEX_Net": "Neutral", "Call_Wall": 0, "Put_Wall": 0, 
        "Vanna_Status": "Neutral", "Charm_Status": "Neutral",
        "Current_Price": 0
    }
    try:
        # 1. 价格与基差
        market_data = yf.download(["NQ=F", "^NDX", "QQQ", "^VIX"], period="2d", progress=False)['Close']
        if isinstance(market_data.columns, pd.MultiIndex): 
            market_data.columns = market_data.columns.get_level_values(0)
        
        fut = market_data['NQ=F'].iloc[-1]
        spot = market_data['^NDX'].iloc[-1]
        qqq_price = market_data['QQQ'].iloc[-1]
        vix_curr = market_data['^VIX'].iloc[-1]
        vix_prev = market_data['^VIX'].iloc[-2]
        
        res['Current_Price'] = qqq_price
        
        basis = fut - spot
        res['Futures_Basis'] = basis
        if basis < -15: res['Basis_Status'] = "🔴 Backwardation (极度看空)"
        elif basis > 60: res['Basis_Status'] = "🟢 Contango (正常)"
        else: res['Basis_Status'] = "⚪ Neutral"
        
        # 2. Gamma 结构 (聚合)
        qqq = yf.Ticker("QQQ")
        expirations = qqq.options[:3] 
        all_calls = []; all_puts = []
        
        for date in expirations:
            try:
                chain = qqq.option_chain(date)
                c = chain.calls.fillna(0); p = chain.puts.fillna(0)
                c = c[c['openInterest'] > 100]; p = p[p['openInterest'] > 100]
                all_calls.append(c[['strike', 'openInterest']])
                all_puts.append(p[['strike', 'openInterest']])
            except: continue
        
        if all_calls and all_puts:
            df_calls = pd.concat(all_calls).groupby('strike')['openInterest'].sum()
            df_puts = pd.concat(all_puts).groupby('strike')['openInterest'].sum()
            res['Call_Wall'] = df_calls.idxmax()
            res['Put_Wall'] = df_puts.idxmax()
            
            # Gamma 判定
            range_min = qqq_price * 0.98; range_max = qqq_price * 1.02
            calls_atm = df_calls[(df_calls.index >= range_min) & (df_calls.index <= range_max)].sum()
            puts_atm = df_puts[(df_puts.index >= range_min) & (df_puts.index <= range_max)].sum()
            gamma_ratio = puts_atm / max(1, calls_atm)
            
            if qqq_price < res['Put_Wall']: res['GEX_Net'] = "🔴 Negative Gamma (高波)"
            elif qqq_price > res['Call_Wall']: res['GEX_Net'] = "🟢 Positive Gamma (突破)"
            else:
                if gamma_ratio > 1.2: res['GEX_Net'] = "🟠 Weak Negative (震荡偏弱)"
                else: res['GEX_Net'] = "🟢 Positive Gamma (震荡偏强)"

        # 3. [新增] Vanna / Charm 代理算法
        # Vanna Logic: 
        # 市场涨 + VIX跌 = Dealers Buy Back Hedges -> Tailwind (助涨)
        # 市场跌 + VIX涨 = Dealers Sell Hedges -> Headwind (助跌)
        ndx_change = spot - market_data['^NDX'].iloc[-2]
        vix_change = vix_curr - vix_prev
        
        if ndx_change > 0 and vix_change < 0:
            res['Vanna_Status'] = "🟢 Tailwind (VIX跌->做市商回补)"
        elif ndx_change < 0 and vix_change > 0:
            res['Vanna_Status'] = "🔴 Headwind (VIX涨->做市商抛售)"
        else:
            res['Vanna_Status'] = "⚪ Neutral (无明显流向)"
            
        # Charm Logic (Time Decay):
        # 接近周五/月底时，时间价值衰减加速。
        # 如果是 Positive Gamma，Dealer Long Option，时间流逝导致 Delta 衰减 -> Dealer 需要卖出 -> 阻力?
        # 通常逻辑: Positive Gamma 下，Charm 倾向于让价格稳定。
        # 简单代理: 看看今天是周几
        weekday = datetime.datetime.now().weekday() # 0=Mon, 4=Fri
        if "Positive" in res['GEX_Net']:
            if weekday >= 3: res['Charm_Status'] = "🟢 Support (时间价值衰减支撑)"
            else: res['Charm_Status'] = "⚪ Neutral"
        else:
            res['Charm_Status'] = "⚪ Neutral (负Gamma不看Charm)"

    except Exception as e: print(f"Deriv Error: {e}")
    return res

@st.cache_data(ttl=1800)
def get_qqq_options_data():
    qqq = yf.Ticker("QQQ")
    res = {"PCR": 0.0, "Unusual": []}
    try:
        expirations = qqq.options[:3]
        total_c_vol = 0; total_p_vol = 0; unusual = []
        for date in expirations:
            try:
                chain = qqq.option_chain(date)
                calls = chain.calls.fillna(0); puts = chain.puts.fillna(0)
                total_c_vol += calls['volume'].sum(); total_p_vol += puts['volume'].sum()
                for opt_type, df, icon in [("CALL", calls, "🟢"), ("PUT", puts, "🔴")]:
                    hot = df[(df['volume'] > 2000) & (df['volume'] > df['openInterest'] * 1.2)]
                    for _, row in hot.iterrows():
                        unusual.append({
                            "Type": f"{icon} {opt_type}", "Strike": row['strike'], "Exp": date,
                            "Vol": int(row['volume']), "OI": int(row['openInterest']),
                            "Ratio": round(row['volume'] / (row['openInterest']+1), 1)
                        })
            except: continue
        if total_c_vol > 0: res['PCR'] = round(total_p_vol / total_c_vol, 2)
        res['Unusual'] = sorted(unusual, key=lambda x: x['Vol'], reverse=True)[:10]
    except: pass
    return res

# 日历
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
    today = datetime.date.today()
    events = []
    next_month = today.replace(day=28) + datetime.timedelta(days=4)
    next_cpi = today.replace(day=12) 
    if today.day > 12: next_cpi = (next_month - datetime.timedelta(days=1)).replace(day=12)
    events.append({"Date": next_cpi, "Event": "CPI (Est)", "Type": "Inflation"})
    events = sorted(events, key=lambda x: x['Date'])
    df = pd.DataFrame(events)
    df = df[df['Date'] >= today].head(5)
    d_df = df.copy(); d_df['Time']="--"; d_df['Est']="--"; d_df['Prev']="--"
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

# --- 2. 核心算法与综述 ---

def calculate_macro_score(ny_fed, fed_liq, credit, rates, vol, opt, deriv, news_score_val):
    score = 0
    flags = [] 
    
    # 1. 流动性 (25%)
    liq_score = 0
    spread = ny_fed['SOFR'] - ny_fed['TGCR']
    if spread > 0.05: liq_score -= 1.0; flags.append("🔴 流动性紧缺 (SOFR > Repo)")
    elif spread < 0.02: liq_score += 0.5
    if fed_liq['RRP_Chg'] > 20: liq_score -= 0.5; flags.append("🔴 RRP 抽水")
    if fed_liq['TGA_Chg'] > 20: liq_score -= 0.5; flags.append("🔴 TGA 抽水")
    if credit[1] < -0.5: liq_score -= 0.5; flags.append("🔴 HYG/LQD 避险模式")
    elif credit[1] > 0.2: liq_score += 0.5
    score += max(-2.5, min(2.5, liq_score))
    
    # 2. 美债 (25%)
    bond_score = 0
    if rates['Yield_10Y'] > 4.5: bond_score -= 1.0; flags.append("🔴 10Y 美债收益率过高")
    elif rates['Yield_10Y'] < 4.0: bond_score += 1.0
    if rates['MOVE'] > 110: bond_score -= 1.5; flags.append("🔴 MOVE 债市恐慌")
    if rates['Inversion'] < -0.5: flags.append("⚠️ 收益率倒挂深度")
    score += max(-2.5, min(2.5, bond_score))
    
    # 3. 恐慌 (15%)
    fear_score = 0
    if vol['VIX'] > 25: fear_score -= 1.0; flags.append("🔴 VIX 恐慌模式")
    elif vol['VIX'] < 13: fear_score -= 0.5
    if vol['Crypto_Val'] < 20: fear_score += 0.5; flags.append("🟢 币圈极度恐慌")
    score += fear_score
    
    # 4. 交易与微观 (20%)
    trade_score = 0
    if opt['PCR'] > 1.2: trade_score -= 0.5; flags.append("📉 PCR 极高 (拥挤)")
    elif opt['PCR'] < 0.6: trade_score += 0.5; flags.append("📈 PCR 极低 (拥挤)")
    if deriv['Basis_Status'].startswith("🔴"): trade_score -= 1.0; flags.append("🔴 期货贴水")
    if "Negative" in deriv['GEX_Net']: trade_score -= 0.5; flags.append("🔴 负 Gamma")
    if "Headwind" in deriv['Vanna_Status']: flags.append("🔴 Vanna 阻力")
    score += max(-2.0, min(2.0, trade_score))
    
    # 5. 新闻 (15%)
    score += news_score_val * 1.5
    
    final_score = round(score * (10 / 7.5), 1)
    summary_text = ""
    action_plan = ""
    if final_score > 3:
        summary_text = "宏观环境**偏多 (Bullish)**，流动性与情绪配合良好。"
        action_plan = "✅ **操作建议**: 逢低做多 (Buy Dips)，关注 Call Wall 阻力位。"
    elif final_score < -3:
        summary_text = "宏观环境**偏空 (Bearish)**，市场面临流动性或恐慌压力。"
        action_plan = "🛡️ **操作建议**: 现金为王，反弹做空，关注 Put Wall 支撑位。"
    else:
        summary_text = "宏观环境**中性震荡 (Neutral)**，多空信号交织。"
        action_plan = "⚖️ **操作建议**: 高抛低吸，避免追涨杀跌，以日内交易为主。"
    if not flags: flags.append("暂无显著异常指标")
    return final_score, flags, summary_text, action_plan

# --- 3. UI ---

with st.spinner("正在聚合全市场数据 (30分钟刷新)..."):
    ai_model = load_ai_model()
    ny_fed = get_ny_fed_data()
    fed_liq = get_fed_liquidity()
    credit = get_credit_spreads()
    rates = get_rates_and_fx()
    vol = get_volatility_indices()
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

    final_score, flags, summary_text, action_plan = calculate_macro_score(ny_fed, fed_liq, credit, rates, vol, opt, deriv, avg_news_score)

# --- HEADER ---
st.title("🦅 QQQ 宏观战情室 Pro (Live)")
current_time = datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M EST')
st.caption(f"上次更新: {current_time} | 自动刷新: 开启 (30分钟)")

summary_class = "summary-bull" if final_score > 3 else "summary-bear" if final_score < -3 else "summary-neutral"
st.markdown(f"""
<div class="summary-box {summary_class}">
    <h3>🛡️ 战情综述 (Score: {final_score})</h3>
    <p style="font-size:1.1em;">{summary_text}</p>
    <p><strong>🚨 异常指标监控:</strong> { '  |  '.join(flags) }</p>
    <hr style="border-top: 1px dashed #ccc;">
    <p style="font-weight:bold;">{action_plan}</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# 1. 流动性
st.subheader("1. 流动性监控 (Liquidity)")
l1, l2, l3, l4, l5 = st.columns(5)
l1.metric("SOFR", f"{ny_fed['SOFR']:.2f}%", f"Spread: {ny_fed['SOFR'] - ny_fed['TGCR']:.3f}")
l2.metric("Repo (TGCR)", f"{ny_fed['TGCR']:.2f}%")
l3.metric("RRP (逆回购)", f"${fed_liq['RRP']:.0f}B", f"{fed_liq['RRP_Chg']:.0f}B", delta_color="inverse")
l4.metric("TGA (财政部)", f"${fed_liq['TGA']:.0f}B", f"{fed_liq['TGA_Chg']:.0f}B", delta_color="inverse")
l5.metric("HYG/LQD", f"{credit[0]:.3f}", f"{credit[1]:.2f}%", help="HYG(高收益)/LQD(投资级)比率。上升代表Risk On，下降代表Risk Off。")

st.divider()

# 2. 美债
st.subheader("2. 美债与汇率 (Rates & FX)")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("10Y 美债收益率", f"{rates['Yield_10Y']:.2f}%", help="全球资产定价之锚。>4.5%利空科技股。")
r2.metric("MOVE (债市恐慌)", f"{rates['MOVE']:.2f}", help="债市波动率。>110 代表极度恐慌。")
r3.metric("2Y/10Y 倒挂", f"{rates['Inversion']:.2f}%", help="经济衰退前瞻。负值越深，衰退概率越大。")
r4.metric("美元指数 (DXY)", f"{rates['DXY']:.2f}")
r5.metric("美元/日元", f"{rates['USDJPY']:.2f}")

st.divider()

# 3. 交易与微观结构
st.subheader("3. 交易与微观结构 (Options & Flows)")
t1, t2, t3, t4 = st.columns(4)
t1.metric("QQQ 期权 PCR", f"{opt['PCR']}", "Put/Call Ratio")
t2.metric("VIX 股市恐慌", f"{vol['VIX']:.2f}")
t3.metric("币圈恐慌指数", f"{vol['Crypto_Val']}", f"{vol['Crypto_Text']}")
t4.metric("期货基差 (Basis)", f"{deriv['Futures_Basis']:.2f}", deriv['Basis_Status'], help="期货-现货。正数正常；负数代表极度恐慌。")

g1, g2, g3, g4 = st.columns(4)
g1.metric("Gamma 状态", deriv['GEX_Net'], help="Positive: 低波动/高抛低吸。Negative: 高波动/追涨杀跌。")
g2.metric("Vanna 流向", deriv['Vanna_Status'], help="Tailwind: VIX跌推升股价。Headwind: VIX涨打压股价。")
g3.metric("Put Wall", f"${deriv['Put_Wall']}", "最大空头Gamma")
g4.metric("Call Wall", f"${deriv['Call_Wall']}", "最大多头Gamma")

with st.expander("查看 QQQ 异动雷达 (Volume > OI)", expanded=True):
    if opt['Unusual']: st.dataframe(pd.DataFrame(opt['Unusual']), use_container_width=True)
    else: st.info("今日无显著异动。")

st.divider()

# 4. 新闻
st.subheader("4. 宏观新闻情报")
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
    # --- [升级版] 模块 6: 日内战术面板 (High Frequency) ---
st.subheader("6. 日内交易战术面板 (0DTE & Micro Structure)")

# 注意：为了保证日内时效性，这里的缓存设为 30秒
# 但前提是你需要手动点击刷新，或者把自动刷新频率调高
@st.cache_data(ttl=30) 
def get_intraday_tactics():
    res = {
        "VWAP": 0, "Price": 0, "Trend": "Neutral",
        "Exp_Move": 0, "Upper_Band": 0, "Lower_Band": 0,
        "0DTE_Call_Vol": 0, "0DTE_Put_Vol": 0, "0DTE_Sentiment": "Neutral",
        "Last_Update": datetime.datetime.now().strftime("%H:%M:%S")
    }
    try:
        # 1. 改为 1分钟 粒度，获取更精准的 VWAP
        df = yf.download("QQQ", period="1d", interval="1m", progress=False)
        
        if not df.empty:
            # 计算 VWAP (Volume Weighted Average Price)
            # 公式: sum(Price * Vol) / sum(Vol)
            # 使用 HLC/3 作为典型价格
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['PV'] = df['Typical_Price'] * df['Volume']
            
            # 累加计算当天的 VWAP
            vwap = df['PV'].sum() / df['Volume'].sum()
            
            current_price = df['Close'].iloc[-1]
            res['VWAP'] = vwap
            res['Price'] = current_price
            
            # 判定乖离率 (0.1% 阈值)
            threshold = 0.001 
            if current_price > vwap * (1 + threshold): 
                res['Trend'] = "🟢 多头强势 (Above VWAP)"
            elif current_price < vwap * (1 - threshold): 
                res['Trend'] = "🔴 空头压制 (Below VWAP)"
            else: 
                res['Trend'] = "⚪ 震荡缠绕 (At VWAP)"
            
        # 2. 计算今日预期波动 (Expected Move)
        # 使用 1分钟 VIX 数据更准
        vix_df = yf.download("^VIX", period="1d", interval="1m", progress=False)
        if not vix_df.empty:
            vix = vix_df['Close'].iloc[-1]
        else:
            vix = 15.0 # 兜底
            
        # 日波动率 ≈ VIX / 16
        daily_vol_pct = (vix / 16) / 100
        exp_move = res['Price'] * daily_vol_pct
        
        res['Exp_Move'] = exp_move
        res['Upper_Band'] = res['Price'] + exp_move
        res['Lower_Band'] = res['Price'] - exp_move
        
        # 3. 0DTE 情绪 (依然受限于 Yahoo 延迟，仅作参考)
        qqq = yf.Ticker("QQQ")
        dates = qqq.options
        target_date = dates[0] 
        chain = qqq.option_chain(target_date)
        
        c_vol = chain.calls['volume'].sum()
        p_vol = chain.puts['volume'].sum()
        
        res['0DTE_Call_Vol'] = c_vol
        res['0DTE_Put_Vol'] = p_vol
        res['Expiry_Date'] = target_date
        
        # 简单的多空比
        ratio = p_vol / c_vol if c_vol > 0 else 1
        if ratio < 0.8: res['0DTE_Sentiment'] = "🟢 Call 主导 (追涨)"
        elif ratio > 1.2: res['0DTE_Sentiment'] = "🔴 Put 主导 (避险)"
        else: res['0DTE_Sentiment'] = "⚪ 多空平衡"

    except Exception as e: pass
    return res

# UI 渲染
with st.spinner("正在计算 1分钟级 VWAP 与 0DTE 数据..."):
    tactics = get_intraday_tactics()

# 显示数据时间戳，提醒时效性
st.caption(f"⚡ 日内数据快照时间: {tactics['Last_Update']} (请手动刷新以获取最新)")

c_day1, c_day2, c_day3, c_day4 = st.columns(4)

# 1. VWAP 趋势
c_day1.metric("日内 VWAP", f"${tactics['VWAP']:.2f}", tactics['Trend'], delta_color="off")

# 2. 预期波动
c_day2.metric("今日预期波动", f"±${tactics['Exp_Move']:.2f}", f"上沿 ${tactics['Upper_Band']:.2f}")

# 3. 0DTE 情绪
c_day3.metric(f"0DTE 情绪 ({tactics.get('Expiry_Date','')})", tactics['0DTE_Sentiment'], f"PCR (Vol): {tactics['0DTE_Put_Vol']/max(1,tactics['0DTE_Call_Vol']):.2f}")

# 4. 现价
c_day4.metric("QQQ 实时价", f"${tactics['Price']:.2f}", f"距离 VWAP: {((tactics['Price']-tactics['VWAP'])/tactics['VWAP'])*100:.2f}%")

with st.expander("🏹 日内期权狙击指南 (Intraday Cheat Sheet)", expanded=True):
    st.markdown(f"""
    *   **判断逻辑**:
        1.  **看 VWAP**: 价格在 VWAP 之上不做空，之下不做多。
        2.  **看边界**: 价格触及 `${tactics['Upper_Band']:.2f}` (预期波动上沿) 时，往往动能耗尽，不要追涨。
        3.  **看 0DTE PCR**: 如果 PCR < 0.7 (极低) 且价格在 VWAP 之下，小心诱多崩盘。
    *   **⚠️ 注意**: Yahoo 数据可能有延迟。**请以你的券商软件报价为准进行下单**，本面板仅用于判断多空风向。
    """)
