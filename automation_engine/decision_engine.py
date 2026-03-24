# ==========================================
# DECISION ENGINE
# ==========================================

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import json

from automation_engine.signals import *
from automation_engine.state_manager import get_state, set_state
from automation_engine.validator import (
    validate_buy_range,
    validate_sell_range,
    should_hold_buy,
    should_hold_sell,
)

from broker_api.broker_factory import get_broker


OUTPUT_DIR = ROOT / "outputs"

FORECAST_FILE = OUTPUT_DIR / "ensemble_forecast.csv"
SETTINGS_FILE = ROOT / "automation_engine" / "user_settings.json"


# ==========================================
# LOAD SETTINGS
# ==========================================

def load_settings():

    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)


# ==========================================
# LOAD FORECAST
# ==========================================

def load_forecast():

    print("Searching forecast in:", OUTPUT_DIR)

    if not FORECAST_FILE.exists():
        print("Ensemble forecast file not found")
        return None

    print("Using forecast:", FORECAST_FILE)

    df = pd.read_csv(FORECAST_FILE)

    return df


# ==========================================
# GET CURRENT PRICE
# ==========================================

def get_current_price(df, metal):

    return df[f"{metal}_Ensemble"].iloc[0]


# ==========================================
# GET PREDICTION SERIES
# ==========================================

def get_prediction_series(df, metal):

    return df[f"{metal}_Ensemble"].values


# ==========================================
# MAIN DECISION FUNCTION
# ==========================================

def run_decision(metal="Gold"):

    broker = get_broker()

    df = load_forecast()

    if df is None:
        return NO_TRADE, "No forecast"

    settings = load_settings()

    state = get_state()

    current_price = get_current_price(df, metal)

    predictions = get_prediction_series(df, metal)

    metal_settings = settings.get(metal)

    if not metal_settings:
        return ERROR, f"Settings for '{metal}' not found"

    buy_price = metal_settings.get("buy_price")
    sell_price = metal_settings.get("sell_price")
    grams = metal_settings.get("grams")

    if buy_price is None or sell_price is None or grams is None:
        return ERROR, "Invalid settings"

    # =====================
    # BUY STATE
    # =====================

    if state == "WAITING_BUY":

        ok, msg = validate_buy_range(current_price, buy_price)

        if not ok:
            return ERROR, msg

        if current_price <= buy_price:

            if should_hold_buy(predictions):
                return HOLD_BUY, "Prediction says wait"

            result, msg = broker.buy(
                metal,
                current_price,
                grams
            )

            if result:
                set_state("WAITING_SELL")
                return BUY, msg

            return ERROR, msg

        return HOLD, "Waiting for buy price"

    # =====================
    # SELL STATE
    # =====================

    if state == "WAITING_SELL":

        ok, msg = validate_sell_range(current_price, sell_price)

        if not ok:
            return ERROR, msg

        if current_price >= sell_price:

            if should_hold_sell(predictions):
                return HOLD_SELL, "Prediction says wait"

            result, msg = broker.sell(
                metal,
                current_price,
                grams
            )

            if result:
                set_state("WAITING_BUY")
                return SELL, msg

            return ERROR, msg

        return HOLD, "Waiting for sell price"

    return NO_TRADE, "No condition met"