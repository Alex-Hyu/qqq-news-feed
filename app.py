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
    # 默认 30分钟刷新宏观，日内数据单独高频刷新
    count = st_autorefresh(interval=30 * 60 * 1000, key="data_refresher")
    st.caption(f"🟢 自动刷新: 开启 (30分钟)")
    if st.button("🔄 立即刷新"):
        st.rerun()

# --- 1. 核心数据获取 (宏观) ---

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
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        ratio = data['HYG'] / data['LQD']
        curr = ratio.iloc[-1]
        pct = ((curr - ratio.iloc[-2]) / ratio.iloc[-2]) * 100
        return curr, pct
    except: return 0, 0

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
    except: pass
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
    except: data['Crypto_Val'] = 50; data['Crypto_Text'] = "Unknown"
    return data

@st.cache_data(ttl=1800)
def get_derivatives_structure():
    res = {
        "Futures_Basis": 0, "Basis_Status": "Normal", 
        "GEX_Net": "Neutral", "Call_Wall": 0, "Put_Wall": 0, 
        "Vanna_Status": "Neutral", "Current_Price": 0
    }
    try:
        market_data = yf.download(["NQ=F", "^NDX", "QQQ", "^VIX"], period="2d", progress=False)['Close']
        if isinstance(market_data.columns, pd.MultiIndex): market_data.columns = market_data.columns.get_level_values(0)
        
        fut = market_data['NQ=F'].iloc[-1]
        spot = market_data['^NDX'].iloc[-1]
        qqq_price = market_data['QQQ'].iloc[-1]
        res['Current_Price'] = qqq_price
        
        basis = fut - spot
        res['Futures_Basis'] = basis
        if basis < -15: res['Basis_Status'] = "🔴 Backwardation"
        elif basis > 60: res['Basis_Status'] = "🟢 Contango"
        else: res['Basis_Status'] = "⚪ Neutral"
        
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
        
        if all_calls:
            df_calls = pd.concat(all_calls).groupby('strike')['openInterest'].sum()
            df_puts = pd.concat(all_puts).groupby('strike')['openInterest'].sum()
            res['Call_Wall'] = df_calls.idxmax()
            res['Put_Wall'] = df_puts.idxmax()
            
            range_min = qqq_price * 0.98; range_max = qqq_price * 1.02
            calls_atm = df_calls[(df_calls.index >= range_min) & (df_calls.index <= range_max)].sum()
            puts_atm = df_puts[(df_puts.index >= range_min) & (df_puts.index <= range_max)].sum()
            gamma_ratio = puts_atm / max(1, calls_atm)
            
            if qqq_price < res['Put_Wall']: res['GEX_Net'] = "🔴 Negative Gamma (High Vol)"
            elif qqq_price > res['Call_Wall']: res['GEX_Net'] = "🟢 Positive Gamma (Breakout)"
            else:
                if gamma_ratio > 1.2: res['GEX_Net'] = "🟠 Weak Negative"
                else: res['GEX_Net'] = "🟢 Positive Gamma"

        ndx_chg = spot - market_data['^NDX'].iloc[-2]
        vix_chg = market_data['^VIX'].iloc[-1] - market_data['^VIX'].iloc[-2]
        if ndx_chg > 0 and vix_chg < 0: res['Vanna_Status'] = "🟢 Tailwind (助涨)"
        elif ndx_chg < 0 and vix_chg > 0: res['Vanna_Status'] = "🔴 Headwind (助跌)"
    except: pass
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

# --- 1.1 日内数据获取 (单独高频刷新) ---
@st.cache_data(ttl=60)
def get_intraday_tactics():
    res = {
        "VWAP": 0, "Price": 0, "Trend": "Neutral",
        "Exp_Move": 0, "Upper_Band": 0, "Lower_Band": 0,
        "0DTE_Call_Vol": 0, "0DTE_Put_Vol": 0, "0DTE_Sentiment": "Neutral",
        "Last_Update": datetime.datetime.now().strftime("%H:%M:%S")
    }
    try:
        # 使用 1分钟 K线
        df = yf.download("QQQ", period="1d", interval="1m", progress=False)
        
        if not df.empty:
            # 处理 MultiIndex
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # 计算 VWAP
            df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['PV'] = df['TP'] * df['Volume']
            vwap = df['PV'].sum() / df['Volume'].sum() if df['Volume'].sum() > 0 else 0
            
            current_price = df['Close'].iloc[-1]
            res['VWAP'] = vwap
            res['Price'] = current_price
            
            if vwap > 0:
                if current_price > vwap * 1.001: res['Trend'] = "🟢 多头强势"
                elif current_price < vwap * 0.999: res['Trend'] = "🔴 空头压制"
                else: res['Trend'] = "⚪ 震荡"
            
        # 预期波动
        vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        exp_move = res['Price'] * ((vix/16)/100)
        res['Exp_Move'] = exp_move
        res['Upper_Band'] = res['Price'] + exp_move
        res['Lower_Band'] = res['Price'] - exp_move
        
        # 0DTE
        qqq = yf.Ticker("QQQ")
        target_date = qqq.options[0]
        chain = qqq.option_chain(target_date)
        c_vol = chain.calls['volume'].sum()
        p_vol = chain.puts['volume'].sum()
        res['0DTE_Call_Vol'] = c_vol
        res['0DTE_Put_Vol'] = p_vol
        
        ratio = p_vol / c_vol if c_vol > 0 else 1
        if ratio < 0.8: res['0DTE_Sentiment'] = "🟢 Call 主导"
        elif ratio > 1.2: res['0DTE_Sentiment'] = "🔴 Put 主导"
        else: res['0DTE_Sentiment'] = "⚪ 平衡"
        
        res['Expiry_Date'] = target_date

    except Exception as e: 
        # 出错时保持默认值 0，避免崩溃
        pass
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
    feeds = [("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258")]
    articles = []
    for src, url in feeds:
        try:
            f = feedparser.parse(url)
            for e in f.entries[:4]: articles.append({"Title": e.title, "Link": e.link, "Source": src})
        except: pass
    return pd.DataFrame(articles)

# --- 2. 核心算法 ---

def calculate_macro_score(ny_fed, fed_liq, credit, rates, vol, opt, deriv, news_score_val):
    score = 0; flags = []
    
    # 流动性
    liq_score = 0
    spread = ny_fed['SOFR'] - ny_fed['TGCR']
    if spread > 0.05: liq_score -= 1.0; flags.append("🔴 SOFR 异常")
    elif spread < 0.02: liq_score += 0.5
    if fed_liq['RRP_Chg'] > 20: liq_score -= 0.5; flags.append("🔴 RRP 抽水")
    if fed_liq['TGA_Chg'] > 20: liq_score -= 0.5; flags.append("🔴 TGA 抽水")
    if credit[1] < -0.5: liq_score -= 0.5; flags.append("🔴 HYG/LQD 避险")
    elif credit[1] > 0.2: liq_score += 0.5
    score += max(-2.5, min(2.5, liq_score))
    
    # 美债
    bond_score = 0
    if rates['Yield_10Y'] > 4.5: bond_score -= 1.0; flags.append("🔴 10Y 收益率过高")
    elif rates['Yield_10Y'] < 4.0: bond_score += 1.0
    if rates['MOVE'] > 110: bond_score -= 1.5; flags.append("🔴 MOVE 恐慌")
    if rates['Inversion'] < -0.5: flags.append("⚠️ 收益率倒挂")
    score += max(-2.5, min(2.5, bond_score))
    
    # 恐慌
    fear_score = 0
    if vol['VIX'] > 25: fear_score -= 1.0; flags.append("🔴 VIX 恐慌")
    elif vol['VIX'] < 13: fear_score -= 0.5; flags.append("⚠️ VIX 过低")
    if vol['Crypto_Val'] < 20: fear_score += 0.5
    score += fear_score
    
    # 交易
    trade_score = 0
    if opt['PCR'] > 1.2: trade_score -= 0.5; flags.append("📉 PCR 极高")
    elif opt['PCR'] < 0.6: trade_score += 0.5; flags.append("📈 PCR 极低")
    if deriv['Basis_Status'].startswith("🔴"): trade_score -= 1.0; flags.append("🔴 期货贴水")
    if "Negative" in deriv['GEX_Net']: trade_score -= 0.5; flags.append("🔴 负 Gamma")
    if "Headwind" in deriv['Vanna_Status']: flags.append("🔴 Vanna 阻力")
    score += max(-2.0, min(2.0, trade_score))
    
    # 新闻
    score += news_score_val * 1.5
    
    final_score = round(score * (10 / 7.5), 1)
    summary = ""
    action = ""
    if final_score > 3:
        summary = "宏观环境**偏多 (Bullish)**，流动性配合。"
        action = "✅ **建议**: 逢低做多，关注 Call Wall。"
    elif final_score < -3:
        summary = "宏观环境**偏空 (Bearish)**，面临压力。"
        action = "🛡️ **建议**: 现金为王，反弹做空。"
    else:
        summary = "宏观环境**中性 (Neutral)**，震荡为主。"
        action = "⚖️ **建议**: 日内高抛低吸。"
    if not flags: flags.append("暂无异常")
    return final_score, flags, summary, action

# --- 3. UI ---

with st.spinner("正在聚合全市场数据..."):
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
    # 日内
    tactics = get_intraday_tactics()

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

    final_score, flags, summary, action = calculate_macro_score(ny_fed, fed_liq, credit, rates, vol, opt, deriv, avg_news_score)

# HEADER
st.title("🦅 QQQ 宏观战情室 Pro (Live)")
current_time = datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%H:%M EST')
st.caption(f"Update: {current_time}")

summary_class = "summary-bull" if final_score > 3 else "summary-bear" if final_score < -3 else "summary-neutral"
st.markdown(f"""
<div class="summary-box {summary_class}">
    <h3>🛡️ 战情综述 (Score: {final_score})</h3>
    <p style="font-size:1.1em;">{summary}</p>
    <p><strong>🚨 监控:</strong> { '  |  '.join(flags) }</p>
    <hr style="border-top: 1px dashed #ccc;">
    <p style="font-weight:bold;">{action}</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# 1. 流动性
st.subheader("1. 流动性监控")
l1, l2, l3, l4, l5 = st.columns(5)
l1.metric("SOFR", f"{ny_fed['SOFR']:.2f}%", f"Spread: {ny_fed['SOFR'] - ny_fed['TGCR']:.3f}")
l2.metric("Repo (TGCR)", f"{ny_fed['TGCR']:.2f}%")
l3.metric("RRP", f"${fed_liq['RRP']:.0f}B", f"{fed_liq['RRP_Chg']:.0f}B", delta_color="inverse")
l4.metric("TGA", f"${fed_liq['TGA']:.0f}B", f"{fed_liq['TGA_Chg']:.0f}B", delta_color="inverse")
l5.metric("HYG/LQD", f"{credit[0]:.3f}", f"{credit[1]:.2f}%", help="Risk On/Off")

st.divider()

# 2. 美债
st.subheader("2. 美债与汇率")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("10Y 收益率", f"{rates['Yield_10Y']:.2f}%")
r2.metric("MOVE", f"{rates['MOVE']:.2f}")
r3.metric("2Y/10Y", f"{rates['Inversion']:.2f}%")
r4.metric("DXY", f"{rates['DXY']:.2f}")
r5.metric("USDJPY", f"{rates['USDJPY']:.2f}")

st.divider()

# 3. 交易结构
st.subheader("3. 交易与微观结构")
t1, t2, t3, t4 = st.columns(4)
t1.metric("PCR", f"{opt['PCR']}", "Put/Call Ratio")
t2.metric("VIX", f"{vol['VIX']:.2f}")
t3.metric("币圈恐慌", f"{vol['Crypto_Val']}", f"{vol['Crypto_Text']}")
t4.metric("基差", f"{deriv['Futures_Basis']:.2f}", deriv['Basis_Status'])

g1, g2, g3, g4 = st.columns(4)
g1.metric("Gamma", deriv['GEX_Net'])
g2.metric("Vanna", deriv['Vanna_Status'])
g3.metric("Put Wall", f"${deriv['Put_Wall']}")
g4.metric("Call Wall", f"${deriv['Call_Wall']}")

with st.expander("查看异动雷达", expanded=True):
    if opt['Unusual']: st.dataframe(pd.DataFrame(opt['Unusual']), use_container_width=True)
    else: st.info("无显著异动")

st.divider()

# 4. 宏观新闻
st.subheader("4. 宏观新闻")
col_news_list, col_news_stat = st.columns([3, 1])
with col_news_list:
    if processed_news:
        for item in processed_news:
            css = "news-card news-bull" if item['Sentiment']=="Bullish" else "news-card news-bear" if item['Sentiment']=="Bearish" else "news-card"
            st.markdown(f"""<div class="{css}"><strong>{item['Sentiment']}</strong> | <a href="{item['Link']}">{item['Title']}</a></div>""", unsafe_allow_html=True)
    else: st.write("暂无新闻")
with col_news_stat: st.metric("情绪分", f"{avg_news_score:.2f}")

st.divider()

# 5. 日历
st.subheader(f"5. 宏观日历 ({cal_source})")
st.dataframe(cal_df, hide_index=True, use_container_width=True)

st.divider()

# 6. 日内战术
st.subheader("6. 日内战术面板 (Intraday)")
st.caption(f"Snapshot: {tactics['Last_Update']}")

c_day1, c_day2, c_day3, c_day4 = st.columns(4)
c_day1.metric("VWAP", f"${tactics['VWAP']:.2f}", tactics['Trend'], delta_color="off")
c_day2.metric("预期波动", f"±${tactics['Exp_Move']:.2f}")
c_day3.metric("0DTE 情绪", tactics['0DTE_Sentiment'])

# [修复] 安全的 Delta 计算
vwap_val = tactics['VWAP']
delta_str = "N/A"
if vwap_val > 0:
    pct = ((tactics['Price'] - vwap_val) / vwap_val) * 100
    delta_str = f"{pct:.2f}% vs VWAP"

c_day4.metric("QQQ 现价", f"${tactics['Price']:.2f}", delta_str)

with st.expander("🏹 日内指南", expanded=True):
    st.write(f"上轨: ${tactics['Upper_Band']:.2f} | 下轨: ${tactics['Lower_Band']:.2f}")
    st.write("策略: 价格 > VWAP 逢低多; 价格 < VWAP 逢高空。")
