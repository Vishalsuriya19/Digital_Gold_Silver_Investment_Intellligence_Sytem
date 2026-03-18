import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

DATA_FILE = Path("gold_silver_data.csv")

START_DATE = "2015-01-01"
TODAY = datetime.today().strftime("%Y-%m-%d")

OUNCE_TO_GRAM = 31.1035


def extract_close(df):

    if df is None or df.empty:
        return pd.Series(dtype="float64")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" in df.columns:
        return df["Close"]

    numeric = df.select_dtypes(include=["float64", "int64"])
    return numeric.iloc[:, 0]


def download_full_history():

    print("Downloading full history...")

    try:
        gold = yf.download("GC=F", start=START_DATE, end=TODAY, progress=True, timeout=30)
        silver = yf.download("SI=F", start=START_DATE, end=TODAY, progress=True, timeout=30)
        usdinr = yf.download("USDINR=X", start=START_DATE, end=TODAY, progress=True, timeout=30)
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise

    gold_close = extract_close(gold)
    silver_close = extract_close(silver)
    usd_close = extract_close(usdinr)

    # ✅ align by index (IMPORTANT FIX)
    df = pd.concat(
        [gold_close, silver_close, usd_close],
        axis=1,
        join="inner"
    )

    df.columns = ["GoldUSD", "SilverUSD", "USDINR"]

    # convert to INR per gram
    df["Gold"] = (df["GoldUSD"] * df["USDINR"]) / OUNCE_TO_GRAM
    df["Silver"] = (df["SilverUSD"] * df["USDINR"]) / OUNCE_TO_GRAM

    df = df[["Gold", "Silver"]]

    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)

    return df


def update_dataset():

    if not DATA_FILE.exists():

        df = download_full_history()

        df.to_csv(DATA_FILE, index=False)

        print("Dataset created")

        return

    df = pd.read_csv(DATA_FILE)

    last_date = pd.to_datetime(df["Date"], format='ISO8601').max()

    next_day = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

    gold = yf.download("GC=F", start=next_day, end=TODAY, progress=True, timeout=30)
    silver = yf.download("SI=F", start=next_day, end=TODAY, progress=True, timeout=30)
    usdinr = yf.download("USDINR=X", start=next_day, end=TODAY, progress=True, timeout=30)

    if gold.empty:
        print("No new data")
        return

    gold_close = extract_close(gold)
    silver_close = extract_close(silver)
    usd_close = extract_close(usdinr)

    new_df = pd.concat(
        [gold_close, silver_close, usd_close],
        axis=1,
        join="inner"
    )

    new_df.columns = ["GoldUSD", "SilverUSD", "USDINR"]

    new_df["Gold"] = (new_df["GoldUSD"] * new_df["USDINR"]) / OUNCE_TO_GRAM
    new_df["Silver"] = (new_df["SilverUSD"] * new_df["USDINR"]) / OUNCE_TO_GRAM

    new_df = new_df[["Gold", "Silver"]]

    new_df.reset_index(inplace=True)
    new_df.rename(columns={"index": "Date"}, inplace=True)

    df = pd.concat([df, new_df])

    df.drop_duplicates("Date", inplace=True)

    df.to_csv(DATA_FILE, index=False)

    print("Dataset updated")


if __name__ == "__main__":
    update_dataset()