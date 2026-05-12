import streamlit as st
import pandas as pd
import yfinance as yf
from fredapi import Fred
import plotly.express as px
import time

# ==============================
# 🔑 CONFIG
# ==============================

FRED_API_KEY = "a8557ef4517aacc94bcc832b7c6254cd"
START_DATE = "2015-01-01"

# ==============================
# 📥 DATA FUNCTIONS
# ==============================

@st.cache_data(ttl=86400)
def get_fx_data(start=START_DATE):
    """Fetch FX data with Yahoo Finance primary and FRED fallback"""

    fx_pairs_yf = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X",
    }

    fx_pairs_fred = {
        "EURUSD": "DEXUSEU",
        "GBPUSD": "DEXUSUK",
        "USDJPY": "DEXJPUS",
    }

    df = pd.DataFrame()

    # ==========================
    # 1️⃣ Yahoo Finance
    # ==========================
    try:
        for name, ticker in fx_pairs_yf.items():
            data = yf.download(
                ticker,
                start=start,
                progress=False,
                auto_adjust=True,
                threads=False,
            )

            if data.empty:
                raise ValueError("Empty Yahoo response")

            df[name] = data["Close"]

        if not df.dropna().empty:
            return df, "Yahoo Finance"

    except Exception as e:
        print(f"Yahoo Finance failed: {e}")

    # ==========================
    # 2️⃣ FRED Fallback
    # ==========================
    fred = Fred(api_key=FRED_API_KEY)
    df = pd.DataFrame()

    for name, code in fx_pairs_fred.items():
        series = fred.get_series(code)
        df[name] = series

    df.index = pd.to_datetime(df.index)
    df = df[df.index >= start]

    return df, "FRED"


@st.cache_data(ttl=86400)
def get_macro_data(start=START_DATE):
    fred = Fred(api_key=FRED_API_KEY)

    series = {
        "CPI": "CPIAUCSL",
        "UNRATE": "UNRATE",
        "FEDRATE": "FEDFUNDS",
    }

    df = pd.DataFrame()

    for name, code in series.items():
        for attempt in range(3):
            try:
                df[name] = fred.get_series(code)
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError(f"Failed to fetch {code}")

    df.index = pd.to_datetime(df.index)
    df = df[df.index >= start]

    return df


# ==============================
# 🔗 ANALYSIS FUNCTIONS
# ==============================

def prepare_data(fx, macro):
    fx_returns = fx.pct_change()

    macro_daily = macro.resample("D").ffill().pct_change()

    df = fx_returns.join(macro_daily, how="inner")

    return df.dropna()


def rolling_correlation(df, fx_pair, macro_var, window):
    return df[fx_pair].rolling(window).corr(df[macro_var])


def lag_correlation(df, fx_pair, macro_var, max_lag=20):
    results = {}
    for lag in range(max_lag + 1):
        results[lag] = df[fx_pair].shift(-lag).corr(df[macro_var])
    return pd.Series(results)


def generate_signals(df, fx_pair, macro_var, window=30, threshold=0.3):
    roll_corr = df[fx_pair].rolling(window).corr(df[macro_var])

    lag_corr = {
        lag: df[fx_pair].shift(-lag).corr(df[macro_var])
        for lag in range(1, 11)
    }

    best_lag = max(lag_corr, key=lag_corr.get)

    macro_change = df[macro_var].diff().shift(best_lag)

    df = df.copy()
    df["Signal"] = "HOLD"

    buy_cond = (roll_corr > threshold) & (macro_change > 0)
    sell_cond = (roll_corr > threshold) & (macro_change < 0)

    df.loc[buy_cond, "Signal"] = "BUY"
    df.loc[sell_cond, "Signal"] = "SELL"

    return df, best_lag


# ==============================
# 🚀 STREAMLIT APP
# ==============================

st.title("📊 FX vs Macro Dashboard")

# Load data
fx, fx_source = get_fx_data()
macro = get_macro_data()

df = prepare_data(fx, macro)

if df.empty:
    st.error("No overlapping FX and macro data available.")
    st.stop()

# ==============================
# 📡 DATA SOURCE INFO
# ==============================

st.caption(f"📡 FX data source: **{fx_source}**")

if fx_source != "Yahoo Finance":
    st.warning("Using FRED FX data due to Yahoo Finance availability issues.")

# ==============================
# 🎛️ SIDEBAR
# ==============================

fx_pair = st.sidebar.selectbox("FX Pair", ["EURUSD", "GBPUSD", "USDJPY"])
macro_var = st.sidebar.selectbox("Macro Variable", ["CPI", "UNRATE", "FEDRATE"])
window = st.sidebar.slider("Rolling Window (days)", 10, 120, 30)
threshold = st.sidebar.slider("Correlation Threshold", 0.0, 1.0, 0.3)

window = min(window, len(df) - 1)

df, signal_lag = generate_signals(df, fx_pair, macro_var, window, threshold)

# ==============================
# 💱 FX PRICE
# ==============================

st.subheader("💱 FX Spot Rate")

fig_fx = px.line(
    fx[fx_pair],
    title=f"{fx_pair} Spot Rate",
)

st.plotly_chart(fig_fx, use_container_width=True)

# ==============================
# 📈 ROLLING CORRELATION
# ==============================

st.subheader("📈 Rolling Correlation")

roll_corr = rolling_correlation(df, fx_pair, macro_var, window)

fig1 = px.line(
    roll_corr,
    title=f"{fx_pair} vs {macro_var} (Rolling {window}d)",
)

st.plotly_chart(fig1, use_container_width=True)

# ==============================
# ⏳ LAG ANALYSIS
# ==============================

st.subheader("⏳ Lag Correlation")

lag_corr = lag_correlation(df, fx_pair, macro_var)

fig2 = px.bar(
    x=lag_corr.index,
    y=lag_corr.values,
    labels={"x": "Lag (days)", "y": "Correlation"},
    title="Lagged Correlation",
)

st.plotly_chart(fig2, use_container_width=True)

# ==============================
# 🔥 HEATMAP
# ==============================

st.subheader("🔥 Correlation Heatmap")

corr_matrix = df[[fx_pair, "CPI", "UNRATE", "FEDRATE"]].corr()

fig3 = px.imshow(
    corr_matrix,
    text_auto=True,
    title="Correlation Matrix",
)

st.plotly_chart(fig3, use_container_width=True)

# ==============================
# 📊 FX vs Macro Overlay
# ==============================

st.subheader("📊 FX vs Macro Overlay")

fig4 = px.line(
    df,
    y=[fx_pair, macro_var],
    title=f"{fx_pair} vs {macro_var}",
)

st.plotly_chart(fig4, use_container_width=True)

# ==============================
# 🧠 INSIGHTS
# ==============================

st.subheader("🧠 Quick Insight")

if roll_corr.dropna().empty:
    st.warning("Insufficient data for rolling correlation.")
else:
    latest_corr = roll_corr.dropna().iloc[-1]
    strongest_lag = lag_corr.dropna().idxmax()

    st.write(f"""
    - **Latest rolling correlation:** {latest_corr:.2f}  
    - **Strongest lag effect:** {strongest_lag} days  
    """)

# ==============================
# 📊 TRADING SIGNALS
# ==============================

st.subheader("📍 Trading Signals")

fig_signal = px.line(
    fx[fx_pair],
    title="FX Price with Trading Signals",
)

buy_signals = df[df["Signal"] == "BUY"]
sell_signals = df[df["Signal"] == "SELL"]

fig_signal.add_scatter(
    x=buy_signals.index,
    y=fx.loc[buy_signals.index, fx_pair],
    mode="markers",
    name="BUY",
    marker=dict(symbol="triangle-up", size=10),
)

fig_signal.add_scatter(
    x=sell_signals.index,
    y=fx.loc[sell_signals.index, fx_pair],
    mode="markers",
    name="SELL",
    marker=dict(symbol="triangle-down", size=10),
)

st.plotly_chart(fig_signal, use_container_width=True)

st.subheader("📌 Signal Summary")

st.write(f"Best macro lag used: **{signal_lag} days**")
st.write(df["Signal"].value_counts())