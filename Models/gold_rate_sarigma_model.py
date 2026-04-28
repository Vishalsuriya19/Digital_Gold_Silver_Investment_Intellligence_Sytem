from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_fetcher import FACTOR_COLUMNS, build_future_factor_frame

DATA_FILE = Path("gold_silver_data.csv")
OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")

FORECAST_DAYS = 30

OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.set_index("Date").asfreq("D")
    df = df.ffill().bfill()

    for column in FACTOR_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0

    return df


def get_exogenous_frame(df: pd.DataFrame) -> pd.DataFrame:
    exog_columns = [
        column for column in FACTOR_COLUMNS
        if column in df.columns and df[column].notna().any()
    ]
    if not exog_columns:
        return pd.DataFrame(index=df.index)

    return df[exog_columns].copy().ffill().bfill()


def train_sarimax(series: pd.Series, exog: pd.DataFrame | None = None):
    model = SARIMAX(
        series,
        exog=exog if exog is not None and not exog.empty else None,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_with_exog(
    model_fit,
    future_exog: pd.DataFrame | None,
    forecast_days: int,
) -> pd.Series:
    forecast = model_fit.get_forecast(
        steps=forecast_days,
        exog=future_exog if future_exog is not None and not future_exog.empty else None,
    )
    return forecast.predicted_mean


def train_and_forecast(forecast_days: int = FORECAST_DAYS) -> pd.DataFrame:
    print("[*] Loading dataset for SARIMAX...")
    df = load_dataset()
    exog = get_exogenous_frame(df)

    future_exog = None
    if not exog.empty:
        future_exog = build_future_factor_frame(exog, forecast_days)

    print("[*] Training Gold SARIMAX...")
    gold_model = train_sarimax(df["Gold"], exog)
    print("[*] Training Silver SARIMAX...")
    silver_model = train_sarimax(df["Silver"], exog)

    gold_bundle = {"model": gold_model, "exog_columns": list(exog.columns)}
    silver_bundle = {"model": silver_model, "exog_columns": list(exog.columns)}

    joblib.dump(gold_bundle, MODELS_DIR / "sarimax_gold.pkl")
    joblib.dump(silver_bundle, MODELS_DIR / "sarimax_silver.pkl")

    # Compatibility with older utilities expecting raw fitted models.
    joblib.dump(gold_model, MODELS_DIR / "sarima_gold.pkl")
    joblib.dump(silver_model, MODELS_DIR / "sarima_silver.pkl")

    print("[*] Forecasting with SARIMAX...")
    gold_pred = forecast_with_exog(gold_model, future_exog, forecast_days)
    silver_pred = forecast_with_exog(silver_model, future_exog, forecast_days)

    if future_exog is not None and not future_exog.empty:
        future_dates = future_exog.index
    else:
        future_dates = pd.date_range(
            start=df.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D",
        )

    forecast_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Gold_SARIMAX": gold_pred.values,
            "Silver_SARIMAX": silver_pred.values,
        }
    )

    legacy_df = forecast_df.rename(
        columns={
            "Gold_SARIMAX": "Gold_Predicted",
            "Silver_SARIMAX": "Silver_Predicted",
        }
    )

    forecast_df.to_csv(OUTPUT_DIR / "sarima_forecast.csv", index=False)
    legacy_df.to_csv(OUTPUT_DIR / "sarima_predictions.csv", index=False)

    print("[OK] SARIMAX forecast saved")
    return forecast_df


def main() -> pd.DataFrame:
    return train_and_forecast()


if __name__ == "__main__":
    main()
