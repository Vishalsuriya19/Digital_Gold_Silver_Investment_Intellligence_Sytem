from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from data_fetcher import FACTOR_COLUMNS

DATA_FILE = Path("gold_silver_data.csv")
DB_FILE = Path("gold_silver_data.db")
OUTPUT_FACTOR_FILE = Path("outputs") / "factor_impact.csv"

MARKET_TABLE = "market_data"
LEGACY_TABLE = "prices"
FACTOR_IMPACT_TABLE = "factor_impact"

MARKET_COLUMNS = ["Date", "Gold", "Silver", *FACTOR_COLUMNS]


def _connect() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_FILE)


def _create_indexes(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{MARKET_TABLE}_date "
        f"ON {MARKET_TABLE}(Date)"
    )
    conn.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{LEGACY_TABLE}_date "
        f"ON {LEGACY_TABLE}(Date)"
    )
    conn.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{FACTOR_IMPACT_TABLE}_key "
        f"ON {FACTOR_IMPACT_TABLE}(date, target, feature_name)"
    )
    conn.commit()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MARKET_TABLE} (
                Date TEXT PRIMARY KEY,
                Gold REAL,
                Silver REAL,
                usd_inr REAL,
                crude_oil_price REAL,
                nifty50 REAL,
                inflation REAL,
                interest_rate REAL,
                bond_yield REAL,
                sentiment_score REAL
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {LEGACY_TABLE} (
                Date TEXT PRIMARY KEY,
                Gold REAL,
                Silver REAL
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {FACTOR_IMPACT_TABLE} (
                date TEXT NOT NULL,
                target TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                impact_percent REAL NOT NULL,
                PRIMARY KEY (date, target, feature_name)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT,
                factors TEXT,
                sentiment REAL,
                url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, title, source)
            )
            """
        )
        _create_indexes(conn)


def normalize_market_data(df: pd.DataFrame) -> pd.DataFrame:
    market_df = df.copy()

    if "Date" not in market_df.columns:
        raise ValueError("Market dataset must contain a 'Date' column.")

    market_df["Date"] = pd.to_datetime(market_df["Date"], errors="coerce")
    market_df = market_df.dropna(subset=["Date"])

    for column in MARKET_COLUMNS:
        if column == "Date":
            continue
        if column not in market_df.columns:
            market_df[column] = pd.NA
        market_df[column] = pd.to_numeric(market_df[column], errors="coerce")

    market_df = market_df.sort_values("Date").drop_duplicates("Date", keep="last")
    market_df["Date"] = market_df["Date"].dt.strftime("%Y-%m-%d")

    return market_df[MARKET_COLUMNS]


def sync_csv_to_db(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is None:
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"CSV file not found: {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)

    market_df = normalize_market_data(df)
    legacy_df = market_df[["Date", "Gold", "Silver"]].copy()

    with _connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {MARKET_TABLE}")
        conn.execute(f"DROP TABLE IF EXISTS {LEGACY_TABLE}")
        conn.commit()

    init_db()
    with _connect() as conn:
        market_df.to_sql(MARKET_TABLE, conn, if_exists="append", index=False)
        legacy_df.to_sql(LEGACY_TABLE, conn, if_exists="append", index=False)
        _create_indexes(conn)

    print(
        f"Synced {len(market_df)} rows into {DB_FILE} "
        f"tables '{MARKET_TABLE}' and '{LEGACY_TABLE}'"
    )
    return market_df


def normalize_factor_impact(df: pd.DataFrame) -> pd.DataFrame:
    factor_df = df.copy()

    required = {"date", "target", "feature_name", "impact_percent"}
    missing = required.difference(factor_df.columns)
    if missing:
        raise ValueError(f"Factor impact data missing columns: {sorted(missing)}")

    factor_df["date"] = pd.to_datetime(factor_df["date"], errors="coerce")
    factor_df = factor_df.dropna(subset=["date", "target", "feature_name"])
    factor_df["target"] = factor_df["target"].astype(str)
    factor_df["feature_name"] = factor_df["feature_name"].astype(str)
    factor_df["impact_percent"] = pd.to_numeric(
        factor_df["impact_percent"], errors="coerce"
    ).fillna(0.0)
    factor_df["date"] = factor_df["date"].dt.strftime("%Y-%m-%d")
    factor_df = factor_df.sort_values(["date", "target", "impact_percent"], ascending=[True, True, False])

    return factor_df[["date", "target", "feature_name", "impact_percent"]]


def store_factor_impact(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is None:
        if not OUTPUT_FACTOR_FILE.exists():
            raise FileNotFoundError(f"Factor impact file not found: {OUTPUT_FACTOR_FILE}")
        df = pd.read_csv(OUTPUT_FACTOR_FILE)

    factor_df = normalize_factor_impact(df)
    init_db()

    with _connect() as conn:
        distinct_pairs = factor_df[["date", "target"]].drop_duplicates()
        for pair in distinct_pairs.itertuples(index=False):
            conn.execute(
                f"DELETE FROM {FACTOR_IMPACT_TABLE} WHERE date = ? AND target = ?",
                (pair.date, pair.target),
            )
        factor_df.to_sql(FACTOR_IMPACT_TABLE, conn, if_exists="append", index=False)
        _create_indexes(conn)

    print(f"Stored {len(factor_df)} factor impact rows into '{FACTOR_IMPACT_TABLE}'")
    return factor_df


def load_latest_factor_impact(target: Optional[str] = None) -> pd.DataFrame:
    init_db()
    with _connect() as conn:
        try:
            df = pd.read_sql_query(
                f"SELECT date, target, feature_name, impact_percent FROM {FACTOR_IMPACT_TABLE}",
                conn,
            )
        except Exception:
            return pd.DataFrame(columns=["date", "target", "feature_name", "impact_percent"])

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    latest_date = df["date"].max()
    df = df[df["date"] == latest_date]

    if target:
        df = df[df["target"].str.lower() == target.lower()]

    return df.sort_values("impact_percent", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    sync_csv_to_db()
    if OUTPUT_FACTOR_FILE.exists():
        store_factor_impact()
