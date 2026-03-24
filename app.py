# ==========================================
# AI GOLD & SILVER INVESTMENT DASHBOARD
# ==========================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

import json

from paper_trading.wallet_manager import load_wallet
from paper_trading.portfolio import show_portfolio

import subprocess
import os

SETTINGS_FILE = Path("automation_engine/user_settings.json")

# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="DGSI Intelligence",
    layout="wide"
)

OUTPUT_DIR = Path("outputs")
DATA_FILE = Path("gold_silver_data.csv")

st.title("Digital Gold & Silver Investment Intelligence" )
st.caption("Automated Prediction Forecast with Ensemble Model • AI Recommendation • Auto Investvestment Engine")



# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    if not DATA_FILE.exists():
        st.error("Dataset not found. Run pipeline first.")
        return None, None

    if not (OUTPUT_DIR / "ensemble_forecast.csv").exists():
        st.error("Ensemble predictions not found. Run pipeline first.")
        return None, None

    historical = pd.read_csv(DATA_FILE)
    forecast = pd.read_csv(OUTPUT_DIR / "ensemble_forecast.csv")

    historical["Date"] = pd.to_datetime(historical["Date"], format='ISO8601')
    forecast["Date"] = pd.to_datetime(forecast["Date"], format='ISO8601')

    return historical, forecast


historical, forecast = load_data()

if historical is None:
    st.stop()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Controls")

metal = st.sidebar.selectbox(
    "Select Metal",
    ["Gold", "Silver"]
)

days = st.sidebar.slider(
    "Forecast Days",
    1,
    30,
    30
)

def load_settings():

    if not SETTINGS_FILE.exists():
        return None

    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)


def save_settings(data):

    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=4)

settings = load_settings()

# ==========================================
# CURRENT PRICE
# ==========================================

latest_price = historical[metal].iloc[-1]

forecast_subset = forecast.head(days)

col1, col2, col3 = st.columns(3)

col1.metric(
    f"{metal} Current Price",
    f"₹ {latest_price:,.2f}"
)

change = (
    forecast_subset[f"{metal}_Ensemble"].iloc[-1]
    - forecast_subset[f"{metal}_Ensemble"].iloc[0]
)

col2.metric(
    "Expected Change",
    f"₹ {change:,.2f}"
)

trend = "UP" if change > 0 else "DOWN" if change < 0 else "STABLE"

col3.metric(
    "Trend",
    trend
)


# ==========================================
# GOLD GRAPH
# ==========================================

if metal == "Gold":
    st.subheader("Gold Price (Last 30 Days + Prediction)")

    gold_hist = historical.tail(30)
    gold_pred = forecast_subset.head(30)

    fig_gold = go.Figure()

    fig_gold.add_trace(
        go.Scatter(
            x=gold_hist["Date"],
            y=gold_hist["Gold"],
            name="Past",
            mode="lines"
        )
    )

    fig_gold.add_trace(
        go.Scatter(
            x=gold_pred["Date"],
            y=gold_pred["Gold_Ensemble"],
            name="Prediction",
            mode="lines+markers"
        )
    )

    fig_gold.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Date",
        yaxis_title="Gold Price",
        xaxis=dict(rangeslider=dict(visible=True))
    )

    st.plotly_chart(fig_gold, use_container_width=True)


# ==========================================
# SILVER GRAPH
# ==========================================

if metal == "Silver":
    st.subheader("Silver Price (Last 30 Days + Prediction)")

    silver_hist = historical.tail(30)
    silver_pred = forecast_subset.head(30)

    fig_silver = go.Figure()

    fig_silver.add_trace(
        go.Scatter(
            x=silver_hist["Date"],
            y=silver_hist["Silver"],
            name="Past",
            mode="lines"
        )
    )

    fig_silver.add_trace(
        go.Scatter(
            x=silver_pred["Date"],
            y=silver_pred["Silver_Ensemble"],
            name="Prediction",
            mode="lines+markers"
        )
    )

    fig_silver.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Date",
        yaxis_title="Silver Price",
        xaxis=dict(rangeslider=dict(visible=True))
    )

    st.plotly_chart(fig_silver, use_container_width=True)


# ==========================================
# TABLE
# ==========================================

st.subheader("Forecast Table")

table_df = forecast_subset[[
    "Date",
    f"{metal}_Ensemble",
    f"{metal}_Lower",
    f"{metal}_Upper"
]].copy()

table_df.columns = [
    "Date",
    "Predicted",
    "Lower",
    "Upper"
]

st.dataframe(table_df, use_container_width=True)


# ==========================================
# AI RECOMMENDATION
# ==========================================

st.subheader("AI Recommendation")

if change > 1:
    st.success("BUY – Uptrend expected")

elif change < -1:
    st.error("SELL – Downtrend expected")

else:
    st.warning("HOLD – Stable market")

# ==========================================
# FULL PIPELINE
# ==========================================

st.subheader("AI Pipeline")

colA, colB = st.columns(2)

if colA.button("Run Full Pipeline"):

    with st.spinner("Running full AI pipeline..."):

        subprocess.run(
            ["python", "master_pipeline.py"]
        )

    st.success("Pipeline completed")


if colB.button("Run Auto Investment Only"):

    with st.spinner("Running auto engine..."):

        subprocess.run(
            ["python", "automation_engine/auto_engine.py"]
        )

    st.success("Auto engine completed")

if st.button("Refresh Dashboard"):

    st.rerun()


# ==========================================
# AUTO INVESTMENT SETTINGS
# ==========================================

st.subheader("Auto Investment Settings")

if settings:

    metal_setting = settings[metal]

    buy_price = st.number_input(
        "AutoBuy price",
        value=float(metal_setting["buy_price"])
    )

    sell_price = st.number_input(
        "AutoSell price",
        value=float(metal_setting["sell_price"])
    )

    grams = st.number_input(
        "Grams",
        value=int(metal_setting["grams"]),
        step=1
    )

    auto_mode = st.toggle("Enable Auto Investment", value=True)

    if st.button("Save Settings"):

        settings[metal]["buy_price"] = buy_price
        settings[metal]["sell_price"] = sell_price
        settings[metal]["grams"] = grams

        save_settings(settings)

        st.success("Settings saved")
# ==========================================
# RUN AUTO ENGINE
# ==========================================

if st.button("Set Auto Investment"):

    import subprocess

    subprocess.run(
        ["python", "automation_engine/auto_engine.py"]
    )

    st.success("Auto engine executed")

# ==========================================
# RUN LIVE ENGINE
# ==========================================
st.subheader("Live Price Engine")

if st.button("Start Live Engine"):

    subprocess.Popen(
        ["python", "price_api/live_price_loop.py"]
    )

    st.success("Live engine started")

import os

if st.button("Stop Live Engine"):

    os.system("taskkill /f /im python.exe")

    st.warning("Live engine stopped")

# ==========================================
# WALLET VIEW
# ==========================================

st.subheader("Trade Wallet")

wallet = load_wallet()

if wallet:

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Balance",
        f"₹ {wallet['balance']:,.2f}"
    )

    col2.metric(
        "Gold grams",
        wallet["gold_grams"]
    )

    col3.metric(
        "Silver grams",
        wallet["silver_grams"]
    )

# ==========================================
# PORTFOLIO VALUE
# ==========================================

if wallet:

    gold_price = forecast[f"Gold_Ensemble"].iloc[0]
    silver_price = forecast[f"Silver_Ensemble"].iloc[0]

    gold_value = wallet["gold_grams"] * gold_price
    silver_value = wallet["silver_grams"] * silver_price

    total = wallet["balance"] + gold_value + silver_value

    st.subheader("Portfolio Value")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Gold Value",
        f"₹ {gold_value:,.2f}"
    )

    c2.metric(
        "Silver Value",
        f"₹ {silver_value:,.2f}"
    )

    c3.metric(
        "Total",
        f"₹ {total:,.2f}"
    )
# ==========================================
# HISTORY
# ==========================================

st.subheader("Trade History")

if wallet:

    history = wallet["history"]

    if len(history) > 0:
        st.dataframe(history, use_container_width=True)
    else:
        st.info("No trades yet")


# ==========================================
# FOOTER
# ==========================================

st.caption(
    "Built with Python • SARIMA • LSTM • XGBoost • ElasticNet • Ensemble Engine • Streamlit"
)