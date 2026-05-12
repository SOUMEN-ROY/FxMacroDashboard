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


# ==============================
# 📥 DATA FUNCTIONS
# ==============================

@st.cache_data
def get_fx_data(start="2015-01-01"):
    fx_pairs = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X"
    }

    df = pd.DataFrame()

    for name, ticker in fx_pairs.items():
        data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        df[name] = data["Close"]

    return df


@st.cache_data
def get_macro_data(start="2015-01-01"):
    fred = Fred(api_key=FRED_API_KEY)

    series = {
        "CPI": "CPIAUCSL",
        "UNRATE": "UNRATE",
        "FEDRATE": "FEDFUNDS"
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
            raise Exception(f"Failed to fetch {code}")

    df.index = pd.to_datetime(df.index)
    df = df[df.index >= start]

    return df


# ==============================
# 🔗 ANALYSIS FUNCTIONS
# ==============================

def prepare_data(fx, macro):
    fx_returns = fx.pct_change()
    macro_daily = macro.resample("D").ffill()
    df = fx_returns.join(macro_daily, how="inner")
    return df.dropna()


def rolling_correlation(df, fx_pair, macro_var, window):
    return df[fx_pair].rolling(window).corr(df[macro_var])


def lag_correlation(df, fx_pair, macro_var, max_lag=20):
    results = {}
    for lag in range(max_lag + 1):
        corr = df[fx_pair].corr(df[macro_var].shift(lag))
        results[lag] = corr
    return pd.Series(results)

def generate_signals(df, fx_pair, macro_var, window=30):

    # Rolling correlation
    roll_corr = df[fx_pair].rolling(window).corr(df[macro_var])

    # Find best lag
    lag_corr = {
        lag: df[fx_pair].corr(df[macro_var].shift(lag))
        for lag in range(1, 11)
    }
    best_lag = max(lag_corr, key=lag_corr.get)

    # Macro change (with lag)
    macro_change = df[macro_var].diff().shift(best_lag)

    signals = []

    for i in range(len(df)):
        if roll_corr.iloc[i] > 0.3:
            if macro_change.iloc[i] > 0:
                signals.append("BUY")
            elif macro_change.iloc[i] < 0:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        else:
            signals.append("HOLD")

    df["Signal"] = signals
    return df, best_lag
   
# ==============================
# 🚀 STREAMLIT APP
# ==============================

st.title("📊 FX vs Macro Dashboard")

# Load data
fx = get_fx_data()
macro = get_macro_data()
df = prepare_data(fx, macro)

# Sidebar
fx_pair = st.sidebar.selectbox("FX Pair", ["EURUSD", "GBPUSD", "USDJPY"])
macro_var = st.sidebar.selectbox("Macro Variable", ["CPI", "UNRATE", "FEDRATE"])
window = st.sidebar.slider("Rolling Window", 10, 120, 30)

df, best_lag = generate_signals(df, fx_pair, macro_var, window)

# ==============================
# 💱 FX PRICE CHART
# ==============================

st.subheader("💱 FX Spot Rate")

fig_fx = px.line(
    fx[fx_pair],
    title=f"{fx_pair} Spot Rate"
)

st.plotly_chart(fig_fx)

# ==============================
# 📈 ROLLING CORRELATION
# ==============================

st.subheader("📈 Rolling Correlation")

roll_corr = rolling_correlation(df, fx_pair, macro_var, window)

fig1 = px.line(
    roll_corr,
    title=f"{fx_pair} vs {macro_var} (Rolling {window}d)"
)

st.plotly_chart(fig1)

# ==============================
# ⏳ LAG ANALYSIS
# ==============================

st.subheader("⏳ Lag Correlation")

lag_corr = lag_correlation(df, fx_pair, macro_var)

fig2 = px.bar(
    x=lag_corr.index,
    y=lag_corr.values,
    labels={"x": "Lag (days)", "y": "Correlation"},
    title="Lagged Correlation"
)

st.plotly_chart(fig2)

# ==============================
# 🔥 HEATMAP
# ==============================

st.subheader("🔥 Correlation Heatmap")

corr_matrix = df[[fx_pair, "CPI", "UNRATE", "FEDRATE"]].corr()

fig3 = px.imshow(
    corr_matrix,
    text_auto=True,
    title="Correlation Matrix"
)

st.plotly_chart(fig3)

# ==============================
# 📊 OVERLAY CHART
# ==============================

st.subheader("📊 FX vs Macro Overlay")

fig4 = px.line(
    df,
    y=[fx_pair, macro_var],
    title=f"{fx_pair} vs {macro_var}"
)

st.plotly_chart(fig4)

# ==============================
# 🧠 INSIGHTS
# ==============================

st.subheader("🧠 Quick Insight")

latest_corr = roll_corr.iloc[-1]
best_lag = lag_corr.idxmax()

st.write(f"""
- Latest rolling correlation: **{latest_corr:.2f}**
- Strongest lag effect at: **{best_lag} days**
""")

st.subheader("📊 Trading Signals")

# Convert signals to numeric markers
signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
df["Signal_num"] = df["Signal"].map(signal_map)

fig_signal = px.line(fx[fx_pair], title="FX Price with Signals")

# Add markers
buy_signals = df[df["Signal"] == "BUY"]
sell_signals = df[df["Signal"] == "SELL"]

fig_signal.add_scatter(
    x=buy_signals.index,
    y=fx.loc[buy_signals.index, fx_pair],
    mode="markers",
    name="BUY",
    marker=dict(symbol="triangle-up", size=10)
)

fig_signal.add_scatter(
    x=sell_signals.index,
    y=fx.loc[sell_signals.index, fx_pair],
    mode="markers",
    name="SELL",
    marker=dict(symbol="triangle-down", size=10)
)

st.plotly_chart(fig_signal)

st.subheader("📌 Signal Summary")

st.write(f"Best lag used: **{best_lag} days**")

signal_counts = df["Signal"].value_counts()
st.write(signal_counts)