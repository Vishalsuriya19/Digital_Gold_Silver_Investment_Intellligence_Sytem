import yfinance as yf
import pandas as pd
from datetime import datetime
import os

DATA_FILE = "gold_silver_data.csv"

OUNCE_TO_GRAM = 31.1035


def fetch_price(symbol):
    data = yf.download(symbol, period="5d", progress=False)
    return float(data["Close"].iloc[-1])


def convert_to_inr_per_gram(symbol):

    usd_price = fetch_price(symbol)
    usd_inr = fetch_price("USDINR=X")

    inr_per_gram = (usd_price * usd_inr) / OUNCE_TO_GRAM

    return round(inr_per_gram, 2)


def update_dataset():

    today = datetime.now().date()

    gold_price = convert_to_inr_per_gram("GC=F")
    silver_price = convert_to_inr_per_gram("SI=F")

    if os.path.exists(DATA_FILE):

        df = pd.read_csv(DATA_FILE)
        df["Date"] = pd.to_datetime(df["Date"], format='ISO8601').dt.date

    else:

        df = pd.DataFrame(columns=["Date", "Gold", "Silver"])

    if today not in df["Date"].values:

        new_row = pd.DataFrame({
            "Date": [today],
            "Gold": [gold_price],
            "Silver": [silver_price]
        })

        df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(DATA_FILE, index=False)

        print("Dataset updated")

    else:

        print("Already updated today")


if __name__ == "__main__":
    update_dataset()