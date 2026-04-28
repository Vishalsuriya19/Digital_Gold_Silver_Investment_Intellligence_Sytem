from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent
FACTOR_DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_START_DATE = "2015-01-01"
OUNCE_TO_GRAM = 31.1035

PRICE_COLUMNS = ["Gold", "Silver"]
FACTOR_COLUMNS = [
    "usd_inr",
    "crude_oil_price",
    "nifty50",
    "inflation",
    "interest_rate",
    "bond_yield",
    "sentiment_score",
]

DIRECT_TICKERS = {
    "GoldUSD": "GC=F",
    "SilverUSD": "SI=F",
    "usd_inr": "USDINR=X",
    "crude_oil_price": "CL=F",
    "nifty50": "^NSEI",
    "interest_rate": "^IRX",
    "bond_yield": "^TNX",
}

CSV_FACTOR_FILES = {
    "inflation": FACTOR_DATA_DIR / "inflation.csv",
    "interest_rate": FACTOR_DATA_DIR / "interest_rate.csv",
    "bond_yield": FACTOR_DATA_DIR / "bond_yield.csv",
    "sentiment_score": FACTOR_DATA_DIR / "sentiment_score.csv",
}


def extract_close(frame: Optional[pd.DataFrame]) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype="float64")

    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" in data.columns:
        return pd.to_numeric(data["Close"], errors="coerce")

    numeric = data.select_dtypes(include=["float64", "float32", "int64", "int32"])
    if numeric.empty:
        return pd.Series(dtype="float64")

    return pd.to_numeric(numeric.iloc[:, 0], errors="coerce")


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None)
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _normalize_series(series: pd.Series, name: str) -> pd.Series:
    out = series.copy()
    out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None)
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    out = pd.to_numeric(out, errors="coerce")
    out.name = name
    return out


def download_close_series(
    ticker: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None,
) -> pd.Series:
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")

    try:
        frame = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            timeout=20,
        )
    except Exception as exc:
        print(f"Warning: unable to download {ticker}: {exc}")
        return pd.Series(dtype="float64")

    return _normalize_series(extract_close(frame), ticker)


def load_factor_csv(
    factor_name: str,
    reference_index: pd.DatetimeIndex,
) -> pd.Series:
    source = CSV_FACTOR_FILES.get(factor_name)
    if source is None or not source.exists():
        return pd.Series(dtype="float64")

    df = pd.read_csv(source)
    if df.empty:
        return pd.Series(dtype="float64")

    possible_value_columns = [
        factor_name,
        "value",
        "Value",
        "close",
        "Close",
    ]
    value_column = next((col for col in possible_value_columns if col in df.columns), None)

    if "Date" not in df.columns or value_column is None:
        raise ValueError(
            f"{source} must contain 'Date' and one of {possible_value_columns}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    series = pd.Series(
        pd.to_numeric(df[value_column], errors="coerce").values,
        index=df["Date"],
        name=factor_name,
    )
    series = _normalize_series(series, factor_name)

    if reference_index.empty:
        return series

    return series.reindex(reference_index).ffill().bfill()


def _zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=numeric.index)
    return (numeric - numeric.mean()) / std


def build_proxy_factor_frame(
    reference_index: pd.DatetimeIndex,
    reference_prices: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if reference_index.empty:
        return pd.DataFrame(columns=FACTOR_COLUMNS)

    index = pd.DatetimeIndex(reference_index).sort_values()
    t = np.linspace(0.0, 1.0, len(index))

    price_frame = pd.DataFrame(index=index)
    if reference_prices is not None and not reference_prices.empty:
        temp = reference_prices.copy()
        if "Date" in temp.columns:
            temp = temp.set_index("Date")
        temp = _to_datetime_index(temp)
        price_frame = temp.reindex(index).ffill().bfill()

    gold_signal = _zscore(np.log(price_frame.get("Gold", pd.Series(1.0, index=index))).rolling(15, min_periods=1).mean())
    silver_signal = _zscore(np.log(price_frame.get("Silver", pd.Series(1.0, index=index))).rolling(15, min_periods=1).mean())

    usd_inr = pd.Series(
        62.0 + 24.0 * t + 1.8 * np.sin(6.0 * np.pi * t) + 1.3 * gold_signal.values,
        index=index,
        name="usd_inr",
    ).clip(lower=58.0, upper=95.0)

    crude_oil = pd.Series(
        58.0 + 16.0 * np.sin(4.5 * np.pi * t + 0.4) + 11.0 * np.cos(7.0 * np.pi * t) + 4.5 * silver_signal.values,
        index=index,
        name="crude_oil_price",
    ).clip(lower=22.0, upper=125.0)

    nifty50 = pd.Series(
        7800.0 + 14500.0 * t + 650.0 * np.sin(5.0 * np.pi * t) + 340.0 * gold_signal.values,
        index=index,
        name="nifty50",
    ).clip(lower=5000.0)

    inflation = pd.Series(
        4.4 + 0.9 * np.sin(5.0 * np.pi * t) + 0.25 * _zscore(crude_oil).values,
        index=index,
        name="inflation",
    ).clip(lower=2.0, upper=9.0)

    interest_rate = pd.Series(
        5.2 + 0.45 * np.sin(3.2 * np.pi * t + 0.8) + 0.35 * _zscore(inflation).values,
        index=index,
        name="interest_rate",
    ).clip(lower=3.5, upper=8.5)

    bond_yield = pd.Series(
        interest_rate.values + 0.55 + 0.20 * np.cos(4.0 * np.pi * t),
        index=index,
        name="bond_yield",
    ).clip(lower=3.8, upper=9.5)

    sentiment_components = pd.concat(
        [
            nifty50.pct_change().fillna(0.0).rename("nifty"),
            (-usd_inr.pct_change().fillna(0.0)).rename("usd"),
            (-crude_oil.pct_change().fillna(0.0)).rename("oil"),
        ],
        axis=1,
    )
    sentiment = sentiment_components.mean(axis=1).rolling(5, min_periods=1).mean()
    sentiment = (_zscore(sentiment).clip(-3.0, 3.0) / 3.0).rename("sentiment_score")

    return pd.concat(
        [
            usd_inr,
            crude_oil,
            nifty50,
            inflation,
            interest_rate,
            bond_yield,
            sentiment,
        ],
        axis=1,
    )


def build_historical_market_frame(
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None,
    fallback_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")

    existing = pd.DataFrame()
    if fallback_df is not None and not fallback_df.empty:
        existing = fallback_df.copy()
        if "Date" in existing.columns:
            existing["Date"] = pd.to_datetime(existing["Date"], errors="coerce")
            existing = existing.dropna(subset=["Date"]).set_index("Date")
        existing = _to_datetime_index(existing)

    downloaded: dict[str, pd.Series] = {}
    for name, ticker in DIRECT_TICKERS.items():
        series = download_close_series(ticker, start_date=start_date, end_date=end_date)
        if not series.empty:
            if name in {"interest_rate", "bond_yield"} and series.max() > 20:
                series = series / 10.0
            downloaded[name] = series.rename(name)

    price_df = pd.DataFrame()
    if {"GoldUSD", "SilverUSD", "usd_inr"}.issubset(downloaded):
        market_prices = pd.concat(
            [downloaded["GoldUSD"], downloaded["SilverUSD"], downloaded["usd_inr"]],
            axis=1,
            join="outer",
        ).sort_index()
        market_prices = market_prices.ffill().bfill()
        market_prices["Gold"] = (market_prices["GoldUSD"] * market_prices["usd_inr"]) / OUNCE_TO_GRAM
        market_prices["Silver"] = (market_prices["SilverUSD"] * market_prices["usd_inr"]) / OUNCE_TO_GRAM
        price_df = market_prices[["Gold", "Silver"]]

    if not existing.empty:
        existing_prices = existing[[col for col in PRICE_COLUMNS if col in existing.columns]]
        if price_df.empty:
            price_df = existing_prices
        else:
            price_df = price_df.combine_first(existing_prices)

    if price_df.empty:
        return pd.DataFrame(columns=["Date", *PRICE_COLUMNS, *FACTOR_COLUMNS])

    reference_index = pd.DatetimeIndex(price_df.index).sort_values()
    factor_df = build_proxy_factor_frame(reference_index, reference_prices=price_df)

    if not existing.empty:
        available_existing = existing.reindex(reference_index)
        overlap = [col for col in FACTOR_COLUMNS if col in available_existing.columns]
        if overlap:
            factor_df.update(available_existing[overlap])

    overlay = pd.DataFrame(index=reference_index)
    for factor_name in ["usd_inr", "crude_oil_price", "nifty50", "interest_rate", "bond_yield"]:
        series = downloaded.get(factor_name)
        if series is not None and not series.empty:
            overlay[factor_name] = series.reindex(reference_index).ffill().bfill()

    for factor_name in ["inflation", "interest_rate", "bond_yield", "sentiment_score"]:
        series = load_factor_csv(factor_name, reference_index)
        if not series.empty:
            overlay[factor_name] = series

    if not overlay.empty:
        factor_df.update(overlay)

    dataset = price_df.reindex(reference_index).join(factor_df, how="left")
    dataset = dataset.ffill().bfill()

    dataset = dataset.reset_index().rename(columns={"index": "Date"})
    dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return dataset[["Date", *PRICE_COLUMNS, *FACTOR_COLUMNS]]


def build_future_factor_frame(
    history_factors: pd.DataFrame,
    forecast_days: int,
) -> pd.DataFrame:
    if history_factors.empty:
        raise ValueError("Factor history is empty; cannot create future exogenous frame.")

    factors = history_factors.copy()
    if "Date" in factors.columns:
        factors = factors.set_index("Date")
    factors = _to_datetime_index(factors)

    future_dates = pd.date_range(
        start=factors.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D",
    )
    future = pd.DataFrame(index=future_dates, columns=factors.columns, dtype="float64")

    for column in factors.columns:
        series = pd.to_numeric(factors[column], errors="coerce").dropna()
        if series.empty:
            future[column] = 0.0
            continue

        recent = series.tail(min(30, len(series)))
        slope = recent.diff().tail(min(7, max(len(recent) - 1, 1))).mean()
        if pd.isna(slope):
            slope = 0.0

        values = series.iloc[-1] + (0.35 * slope * np.arange(1, forecast_days + 1))
        if column == "sentiment_score":
            values = np.clip(values, -1.0, 1.0)
        else:
            values = np.clip(values, 0.0, None)

        future[column] = values

    return future


def fetch_latest_market_snapshot() -> Optional[dict]:
    fallback_df = None
    local_data_file = PROJECT_ROOT / "gold_silver_data.csv"
    if local_data_file.exists():
        fallback_df = pd.read_csv(local_data_file)

    market_df = build_historical_market_frame(
        start_date=DEFAULT_START_DATE,
        fallback_df=fallback_df,
    )
    if market_df.empty:
        return None

    latest = market_df.iloc[-1]
    return latest.to_dict()


def fetch_today_price() -> Optional[dict]:
    snapshot = fetch_latest_market_snapshot()
    if snapshot is None:
        return None

    return {
        "date": snapshot["Date"],
        "gold": round(float(snapshot["Gold"]), 2),
        "silver": round(float(snapshot["Silver"]), 2),
        "usd_inr": round(float(snapshot["usd_inr"]), 4),
        "crude_oil_price": round(float(snapshot["crude_oil_price"]), 4),
        "nifty50": round(float(snapshot["nifty50"]), 4),
        "inflation": round(float(snapshot["inflation"]), 4),
        "interest_rate": round(float(snapshot["interest_rate"]), 4),
        "bond_yield": round(float(snapshot["bond_yield"]), 4),
        "sentiment_score": round(float(snapshot["sentiment_score"]), 4),
    }


if __name__ == "__main__":
    print(fetch_today_price())
