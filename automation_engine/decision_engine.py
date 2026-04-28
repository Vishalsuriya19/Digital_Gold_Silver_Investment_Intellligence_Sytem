# ==========================================
# DECISION ENGINE
# ==========================================

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from automation_engine.signals import *
from automation_engine.state_manager import get_state, set_state
from automation_engine.validator import (
    should_hold_buy,
    should_hold_sell,
    validate_buy_range,
    validate_sell_range,
)
from broker_api.broker_factory import get_broker

try:
    from sync_csv_to_sqlite import load_latest_factor_impact
except Exception:
    load_latest_factor_impact = None

OUTPUT_DIR = ROOT / "outputs"

FORECAST_FILE = OUTPUT_DIR / "ensemble_forecast.csv"
FACTOR_IMPACT_FILE = OUTPUT_DIR / "factor_impact.csv"
DATA_FILE = ROOT / "gold_silver_data.csv"
SETTINGS_FILE = ROOT / "automation_engine" / "user_settings.json"

FACTOR_DIRECTION = {
    "Gold": {
        "usd_inr": 1,
        "crude_oil_price": 1,
        "nifty50": -1,
        "inflation": 1,
        "interest_rate": -1,
        "bond_yield": -1,
        "sentiment_score": -1,
    },
    "Silver": {
        "usd_inr": 1,
        "crude_oil_price": 1,
        "nifty50": 1,
        "inflation": 1,
        "interest_rate": -1,
        "bond_yield": -1,
        "sentiment_score": 1,
    },
}

HIGH_IMPACT_THRESHOLD = 12.0
STRONG_IMPACT_TOTAL = 30.0


def load_settings():
    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)


def load_forecast():
    print("Searching forecast in:", OUTPUT_DIR)

    if not FORECAST_FILE.exists():
        print("Ensemble forecast file not found")
        return None

    print("Using forecast:", FORECAST_FILE)
    return pd.read_csv(FORECAST_FILE)


def load_market_data():
    if not DATA_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_FILE)
    if "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df


def load_factor_impact(metal="Gold"):
    if FACTOR_IMPACT_FILE.exists():
        df = pd.read_csv(FACTOR_IMPACT_FILE)
    elif load_latest_factor_impact is not None:
        df = load_latest_factor_impact(target=metal)
    else:
        return pd.DataFrame()

    if df.empty:
        return df

    if "target" in df.columns:
        df = df[df["target"].astype(str).str.lower() == metal.lower()]

    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        latest_date = df["date"].max()
        df = df[df["date"] == latest_date]

    return df.sort_values("impact_percent", ascending=False).reset_index(drop=True)


def get_current_price(df, metal):
    return df[f"{metal}_Ensemble"].iloc[0]


def get_prediction_series(df, metal):
    return df[f"{metal}_Ensemble"].values


def _top_factor_summary(factor_df: pd.DataFrame) -> str:
    if factor_df.empty:
        return "Factor impact unavailable."

    top_rows = factor_df.head(3)
    summary = ", ".join(
        f"{row.feature_name} {row.impact_percent:.1f}%"
        for row in top_rows.itertuples(index=False)
    )
    return f"Top factors: {summary}."


def evaluate_factor_context(metal: str, predictions) -> dict:
    market_df = load_market_data()
    factor_df = load_factor_impact(metal)

    context = {
        "strong_buy": False,
        "strong_sell": False,
        "summary": _top_factor_summary(factor_df),
    }

    if factor_df.empty or len(market_df) < 2:
        return context

    latest = market_df.iloc[-1]
    previous = market_df.iloc[-2]
    prediction_change = float(predictions[-1] - predictions[0])

    if prediction_change == 0:
        return context

    direction_map = FACTOR_DIRECTION.get(metal, {})
    bullish_hits = []
    bearish_hits = []
    bullish_weight = 0.0
    bearish_weight = 0.0

    for row in factor_df.itertuples(index=False):
        feature_name = row.feature_name
        impact_percent = float(row.impact_percent)

        if impact_percent < HIGH_IMPACT_THRESHOLD:
            continue
        if feature_name not in latest.index or feature_name not in previous.index:
            continue
        if feature_name not in direction_map:
            continue

        delta = float(latest[feature_name]) - float(previous[feature_name])
        directional_bias = direction_map[feature_name]

        if directional_bias * delta > 0:
            bullish_hits.append(feature_name)
            bullish_weight += impact_percent
        elif directional_bias * delta < 0:
            bearish_hits.append(feature_name)
            bearish_weight += impact_percent

    context["strong_buy"] = prediction_change > 0 and (
        len(bullish_hits) >= 2 or bullish_weight >= STRONG_IMPACT_TOTAL
    )
    context["strong_sell"] = prediction_change < 0 and (
        len(bearish_hits) >= 2 or bearish_weight >= STRONG_IMPACT_TOTAL
    )

    if context["strong_buy"]:
        context["summary"] = (
            f"{context['summary']} Factor context supports a STRONG BUY "
            f"({', '.join(bullish_hits[:3])})."
        )
    elif context["strong_sell"]:
        context["summary"] = (
            f"{context['summary']} Factor context supports a STRONG SELL "
            f"({', '.join(bearish_hits[:3])})."
        )

    return context


def run_decision(metal="Gold"):
    broker = get_broker()
    df = load_forecast()

    if df is None:
        return NO_TRADE, "No forecast"

    settings = load_settings()
    state = get_state()

    current_price = get_current_price(df, metal)
    predictions = get_prediction_series(df, metal)
    factor_context = evaluate_factor_context(metal, predictions)

    metal_settings = settings.get(metal)
    if not metal_settings:
        return ERROR, f"Settings for '{metal}' not found"

    buy_price = metal_settings.get("buy_price")
    sell_price = metal_settings.get("sell_price")
    grams = metal_settings.get("grams")

    if buy_price is None or sell_price is None or grams is None:
        return ERROR, "Invalid settings"

    if state == "WAITING_BUY":
        ok, msg = validate_buy_range(current_price, buy_price)
        if not ok:
            return ERROR, msg

        if current_price <= buy_price:
            if should_hold_buy(predictions) and not factor_context["strong_buy"]:
                return HOLD_BUY, f"Prediction says wait. {factor_context['summary']}"

            result, msg = broker.buy(metal, current_price, grams)
            if result:
                set_state("WAITING_SELL")
                signal = STRONG_BUY if factor_context["strong_buy"] else BUY
                return signal, f"{msg}. {factor_context['summary']}"

            return ERROR, msg

        return HOLD, f"Waiting for buy price. {factor_context['summary']}"

    if state == "WAITING_SELL":
        ok, msg = validate_sell_range(current_price, sell_price)
        if not ok:
            return ERROR, msg

        if current_price >= sell_price:
            if should_hold_sell(predictions) and not factor_context["strong_sell"]:
                return HOLD_SELL, f"Prediction says wait. {factor_context['summary']}"

            result, msg = broker.sell(metal, current_price, grams)
            if result:
                set_state("WAITING_BUY")
                signal = STRONG_SELL if factor_context["strong_sell"] else SELL
                return signal, f"{msg}. {factor_context['summary']}"

            return ERROR, msg

        return HOLD, f"Waiting for sell price. {factor_context['summary']}"

    return NO_TRADE, f"No condition met. {factor_context['summary']}"
