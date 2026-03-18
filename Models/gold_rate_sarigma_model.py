# =====================================
# SARIMA Gold & Silver Forecast (FIXED)
# Uses local INR dataset
# =====================================

import pandas as pd
import joblib
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_FILE = Path("gold_silver_data.csv")
OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")

FORECAST_DAYS = 30

OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------
# Load dataset
# ---------------------------
def load_dataset():

    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"], format='ISO8601')

    df = df.sort_values("Date")

    df.set_index("Date", inplace=True)

    return df


# ---------------------------
# Train SARIMA
# ---------------------------
def train_sarima(series):

    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    model_fit = model.fit(disp=False)

    return model_fit


# ---------------------------
# Forecast
# ---------------------------
def forecast(model):

    forecast = model.get_forecast(steps=FORECAST_DAYS)

    return forecast.predicted_mean


# ---------------------------
# MAIN
# ---------------------------
def main():

    print("[*] Loading dataset...")

    df = load_dataset()

    gold = df["Gold"]
    silver = df["Silver"]

    print("[*] Training SARIMA...")

    gold_model = train_sarima(gold)
    silver_model = train_sarima(silver)

    joblib.dump(gold_model, MODELS_DIR / "sarima_gold.pkl")
    joblib.dump(silver_model, MODELS_DIR / "sarima_silver.pkl")

    print("[*] Forecasting...")

    gold_pred = forecast(gold_model)
    silver_pred = forecast(silver_model)

    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=FORECAST_DAYS,
        freq="D",
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Gold_SARIMA": gold_pred.values,
        "Silver_SARIMA": silver_pred.values,
    })

    forecast_df.to_csv(OUTPUT_DIR / "sarima_forecast.csv", index=False)

    print("[OK] SARIMA forecast saved")


if __name__ == "__main__":
    main()