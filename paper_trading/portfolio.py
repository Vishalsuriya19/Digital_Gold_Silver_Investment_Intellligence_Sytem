import pandas as pd
from pathlib import Path
import json

WALLET_FILE = Path("paper_trading/wallet.json")
FORECAST_FILE = Path("outputs/ensemble_forecast.csv")


def load_wallet():

    with open(WALLET_FILE, "r") as f:
        return json.load(f)


def get_prices():

    df = pd.read_csv(FORECAST_FILE)

    gold = df["Gold_Ensemble"].iloc[0]
    silver = df["Silver_Ensemble"].iloc[0]

    return gold, silver


def show_portfolio():

    wallet = load_wallet()

    gold_price, silver_price = get_prices()

    gold_value = wallet["gold_grams"] * gold_price
    silver_value = wallet["silver_grams"] * silver_price

    total = wallet["balance"] + gold_value + silver_value

    print("\n===== PORTFOLIO =====")

    print("Balance:", wallet["balance"])
    print("Gold value:", gold_value)
    print("Silver value:", silver_value)

    print("TOTAL VALUE:", total)