# ==========================================
# AI GOLD & SILVER INVESTMENT DASHBOARD
# ==========================================

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from paper_trading.wallet_manager import load_wallet

SETTINGS_FILE = Path("automation_engine/user_settings.json")
OUTPUT_DIR = Path("outputs")
DATA_FILE = Path("gold_silver_data.csv")
FACTOR_IMPACT_FILE = OUTPUT_DIR / "factor_impact.csv"

st.set_page_config(page_title="DGSI Intelligence", layout="wide")

st.title("Digital Gold & Silver Investment Intelligence")
st.caption(
    "Automated Forecasting, Explainable AI, Ensemble Predictions, and Auto Investment"
)


def load_data():
    if not DATA_FILE.exists():
        st.error("Dataset not found. Run pipeline first.")
        return None, None

    forecast_file = OUTPUT_DIR / "ensemble_forecast.csv"
    if not forecast_file.exists():
        st.error("Ensemble predictions not found. Run pipeline first.")
        return None, None

    historical = pd.read_csv(DATA_FILE)
    forecast = pd.read_csv(forecast_file)

    historical["Date"] = pd.to_datetime(historical["Date"], errors="coerce")
    forecast["Date"] = pd.to_datetime(forecast["Date"], errors="coerce")

    historical = historical.dropna(subset=["Date"]).sort_values("Date")
    forecast = forecast.dropna(subset=["Date"]).sort_values("Date")

    return historical, forecast


def load_factor_impact():
    if not FACTOR_IMPACT_FILE.exists():
        return pd.DataFrame()

    factor_df = pd.read_csv(FACTOR_IMPACT_FILE)
    if factor_df.empty:
        return factor_df

    factor_df["date"] = pd.to_datetime(factor_df["date"], errors="coerce")
    factor_df = factor_df.dropna(subset=["date"])
    return factor_df


def load_settings():
    if not SETTINGS_FILE.exists():
        return None

    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)


def save_settings(data):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=4)


def run_python_script(script_path: str):
    subprocess.run([sys.executable, script_path], check=False)


historical, forecast = load_data()
if historical is None or forecast is None:
    st.stop()

factor_impact = load_factor_impact()
settings = load_settings()

st.sidebar.header("Controls")
metal = st.sidebar.selectbox("Select Metal", ["Gold", "Silver"])
days = st.sidebar.slider("Forecast Days", 1, 30, 30)

latest_price = historical[metal].iloc[-1]
forecast_subset = forecast.head(days)

col1, col2, col3 = st.columns(3)

col1.metric(f"{metal} Current Price", f"Rs {latest_price:,.2f}")

change = (
    forecast_subset[f"{metal}_Ensemble"].iloc[-1]
    - forecast_subset[f"{metal}_Ensemble"].iloc[0]
)

col2.metric("Expected Change", f"Rs {change:,.2f}")
trend = "UP" if change > 0 else "DOWN" if change < 0 else "STABLE"
col3.metric("Trend", trend)

if metal == "Gold":
    st.subheader("Gold Price (Last 30 Days + Prediction)")
    history_slice = historical.tail(30)
    forecast_slice = forecast_subset.head(30)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=history_slice["Date"], y=history_slice["Gold"], name="Past", mode="lines")
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_slice["Date"],
            y=forecast_slice["Gold_Ensemble"],
            name="Prediction",
            mode="lines+markers",
        )
    )
else:
    st.subheader("Silver Price (Last 30 Days + Prediction)")
    history_slice = historical.tail(30)
    forecast_slice = forecast_subset.head(30)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=history_slice["Date"], y=history_slice["Silver"], name="Past", mode="lines")
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_slice["Date"],
            y=forecast_slice["Silver_Ensemble"],
            name="Prediction",
            mode="lines+markers",
        )
    )

figure.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_title="Date",
    yaxis_title=f"{metal} Price",
    xaxis=dict(rangeslider=dict(visible=True)),
)
st.plotly_chart(figure, use_container_width=True)

st.subheader("Forecast Table")
table_df = forecast_subset[
    ["Date", f"{metal}_Ensemble", f"{metal}_Lower", f"{metal}_Upper"]
].copy()
table_df.columns = ["Date", "Predicted", "Lower", "Upper"]
st.dataframe(table_df, use_container_width=True)

st.subheader("AI Recommendation")
if change > 1:
    st.success("BUY - Uptrend expected")
elif change < -1:
    st.error("SELL - Downtrend expected")
else:
    st.warning("HOLD - Stable market")

st.subheader("Factor Impact Analysis")
if factor_impact.empty:
    st.info("Factor impact output not found. Run the full pipeline to generate SHAP analysis.")
else:
    metal_impact = factor_impact[
        factor_impact["target"].astype(str).str.lower() == metal.lower()
    ].copy()

    if metal_impact.empty:
        st.info(f"No factor impact snapshot available for {metal}.")
    else:
        latest_impact_date = metal_impact["date"].max()
        metal_impact = metal_impact[metal_impact["date"] == latest_impact_date]
        metal_impact = metal_impact.sort_values("impact_percent", ascending=False)
        top_factors = metal_impact.head(6).copy()

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric(
            "Top Factor",
            top_factors.iloc[0]["feature_name"],
            f"{top_factors.iloc[0]['impact_percent']:.2f}%",
        )
        metric_col2.metric(
            "Top 3 Contribution",
            f"{top_factors.head(3)['impact_percent'].sum():.2f}%",
        )

        chart_col1, chart_col2 = st.columns([1, 2])
        chart_col1.dataframe(
            top_factors[["feature_name", "impact_percent"]].rename(
                columns={
                    "feature_name": "Factor",
                    "impact_percent": "Impact %",
                }
            ),
            use_container_width=True,
        )

        bar_chart = px.bar(
            top_factors.sort_values("impact_percent"),
            x="impact_percent",
            y="feature_name",
            orientation="h",
            text="impact_percent",
            labels={"impact_percent": "Impact %", "feature_name": "Factor"},
            template="plotly_dark",
        )
        bar_chart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        bar_chart.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
        chart_col2.plotly_chart(bar_chart, use_container_width=True)
        st.caption(f"Latest SHAP factor snapshot: {latest_impact_date.date()}")

st.subheader("AI Pipeline")
pipeline_col1, pipeline_col2 = st.columns(2)

if pipeline_col1.button("Run Full Pipeline"):
    with st.spinner("Running full AI pipeline..."):
        run_python_script("master_pipeline.py")
    st.success("Pipeline completed")

if pipeline_col2.button("Run Auto Investment Only"):
    with st.spinner("Running auto engine..."):
        run_python_script("automation_engine/auto_engine.py")
    st.success("Auto engine completed")

if st.button("Refresh Dashboard"):
    st.rerun()

st.subheader("Auto Investment Settings")
if settings:
    metal_setting = settings[metal]

    buy_price = st.number_input("AutoBuy price", value=float(metal_setting["buy_price"]))
    sell_price = st.number_input("AutoSell price", value=float(metal_setting["sell_price"]))
    grams = st.number_input("Grams", value=int(metal_setting["grams"]), step=1)
    st.toggle("Enable Auto Investment", value=True, disabled=True)

    if st.button("Save Settings"):
        settings[metal]["buy_price"] = buy_price
        settings[metal]["sell_price"] = sell_price
        settings[metal]["grams"] = grams
        save_settings(settings)
        st.success("Settings saved")
else:
    st.info("Auto investment settings file not found.")

if st.button("Set Auto Investment"):
    run_python_script("automation_engine/auto_engine.py")
    st.success("Auto engine executed")


st.subheader("Latest Market News")

# Display latest news directly from the database (no pipeline instantiation at page load)
import sqlite3
from pathlib import Path

DB_PATH = Path("gold_silver_data.db")

def _get_latest_news_from_db(limit: int = 10):
    if not DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute(
            """
            SELECT date, title, source, factors, sentiment, url
            FROM news_data
            ORDER BY date DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()

        articles = []
        for r in rows:
            articles.append(
                {
                    "date": r[0],
                    "title": r[1],
                    "source": r[2],
                    "factors": json.loads(r[3]) if r[3] else [],
                    "sentiment": r[4],
                    "url": r[5],
                }
            )
        return articles
    except Exception:
        return []


latest_news = _get_latest_news_from_db(limit=10)

if latest_news:
    for article in latest_news:
        st.subheader(article["title"])

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**Source:** {article['source']}")

        with col2:
            sentiment = article.get("sentiment", 0.0)
            if sentiment > 0.1:
                st.write(f"**Sentiment:** 🟢 Positive ({sentiment})")
            elif sentiment < -0.1:
                st.write(f"**Sentiment:** 🔴 Negative ({sentiment})")
            else:
                st.write(f"**Sentiment:** 🟡 Neutral ({sentiment})")

        with col3:
            factors = article.get("factors", [])
            if factors:
                st.write(f"**Factors:** {', '.join(factors)}")

        if article.get("url"):
            st.markdown(f"[Read Full Article]({article['url']})")

        st.markdown("---")
else:
    st.info("No news articles found. Use 'Fetch Latest News' to populate the news table.")


# News Intelligence Controls (lazy pipeline instantiation)
st.subheader("News Intelligence Controls")
news_col1, news_col2 = st.columns(2)

if news_col1.button("Fetch Latest News"):
    with st.spinner("Fetching and processing latest news..."):
        try:
            # Instantiate pipeline only when user requests a fetch
            from news_pipeline import NewsPipeline

            pipeline = NewsPipeline()
            result = pipeline.run_pipeline(max_results=20)
            if result.get("success"):
                st.success(f"Successfully fetched and processed {result.get('articles_saved', 0)} news articles!")
                st.rerun()
            else:
                st.error(f"Failed to fetch news: {result.get('message', 'Unknown error')}")
        except Exception as e:
            st.error(f"News pipeline failed: {e}")
            st.info("Ensure 'gnews' and 'textblob' are installed into the Python used by Streamlit, then restart the app.")

if news_col2.button("Refresh News Display"):
    st.rerun()

st.caption(
    "Built with Python, SARIMAX, LSTM, XGBoost, ElasticNet, Ensemble Forecasting, SHAP, and Streamlit"
)
