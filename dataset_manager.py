from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from data_fetcher import FACTOR_COLUMNS, build_historical_market_frame

DATA_FILE = Path("gold_silver_data.csv")
START_DATE = "2015-01-01"

DATASET_COLUMNS = ["Date", "Gold", "Silver", *FACTOR_COLUMNS]


def load_local_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return pd.DataFrame(columns=DATASET_COLUMNS)

    df = pd.read_csv(DATA_FILE)
    if df.empty:
        return pd.DataFrame(columns=DATASET_COLUMNS)

    return normalize_dataset(df)


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    dataset = df.copy()

    if "Date" not in dataset.columns:
        raise ValueError("Dataset must contain a 'Date' column.")

    dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce")
    dataset = dataset.dropna(subset=["Date"])

    for column in DATASET_COLUMNS:
        if column == "Date":
            continue
        if column not in dataset.columns:
            dataset[column] = pd.NA
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    dataset = dataset.sort_values("Date").drop_duplicates("Date", keep="last")
    dataset = dataset.ffill().bfill()
    dataset["Date"] = dataset["Date"].dt.strftime("%Y-%m-%d")

    return dataset[DATASET_COLUMNS]


def merge_datasets(local_df: pd.DataFrame, refreshed_df: pd.DataFrame) -> pd.DataFrame:
    if local_df.empty:
        return normalize_dataset(refreshed_df)
    if refreshed_df.empty:
        return normalize_dataset(local_df)

    combined = pd.concat([local_df, refreshed_df], ignore_index=True, sort=False)
    return normalize_dataset(combined)


def save_dataset(df: pd.DataFrame, sync_sqlite: bool = True) -> pd.DataFrame:
    dataset = normalize_dataset(df)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATA_FILE, index=False)

    if sync_sqlite:
        try:
            from sync_csv_to_sqlite import sync_csv_to_db

            sync_csv_to_db(dataset)
        except Exception as exc:
            print(f"Warning: SQLite sync skipped: {exc}")

    return dataset


def refresh_dataset() -> pd.DataFrame:
    existing_df = load_local_dataset()
    refreshed_df = build_historical_market_frame(
        start_date=START_DATE,
        end_date=datetime.today().strftime("%Y-%m-%d"),
        fallback_df=existing_df,
    )

    if refreshed_df.empty and existing_df.empty:
        raise RuntimeError("Unable to build dataset from remote or local sources.")

    combined = merge_datasets(existing_df, refreshed_df)
    return save_dataset(combined)


def update_dataset() -> pd.DataFrame:
    dataset_before = load_local_dataset()
    dataset_after = refresh_dataset()

    action = "created" if dataset_before.empty else "updated"
    print(
        f"Dataset {action}: {len(dataset_after)} rows through "
        f"{dataset_after['Date'].iloc[-1]}"
    )

    return dataset_after


if __name__ == "__main__":
    update_dataset()
