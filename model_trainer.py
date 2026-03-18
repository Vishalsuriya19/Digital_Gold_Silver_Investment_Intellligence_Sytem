# =====================================
# MODEL TRAINER - SARIMA (1g Digital Gold & Silver)
# =====================================

import pandas as pd
import joblib
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_PATH = Path("gold_silver_data.csv")
MODEL_DIR = Path("Models")
OUTPUT_DIR = Path("Outputs")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FORECAST_DAYS = 30


def train_and_forecast():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"], format='ISO8601')
    df.set_index("Date", inplace=True)

    df = df.asfreq("D")
    df.ffill(inplace=True)

    # =========================
    # GOLD MODEL
    # =========================
    print("Training Gold SARIMA model...")

    gold_model = SARIMAX(
        df["Gold"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    gold_forecast = gold_model.get_forecast(steps=FORECAST_DAYS)
    gold_pred = gold_forecast.predicted_mean

    joblib.dump(gold_model, MODEL_DIR / "sarima_gold.pkl")

    # =========================
    # SILVER MODEL
    # =========================
    print("Training Silver SARIMA model...")

    silver_model = SARIMAX(
        df["Silver"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    silver_forecast = silver_model.get_forecast(steps=FORECAST_DAYS)
    silver_pred = silver_forecast.predicted_mean

    joblib.dump(silver_model, MODEL_DIR / "sarima_silver.pkl")

    # =========================
    # SAVE PREDICTIONS
    # =========================
    predictions = pd.DataFrame({
        "Date": gold_pred.index,
        "Gold_Predicted": gold_pred.values,
        "Silver_Predicted": silver_pred.values
    })

    # Save predictions in both expected formats for compatibility
    predictions.to_csv(OUTPUT_DIR / "sarima_predictions.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "sarima_forecast.csv", index=False)

    print("SARIMA training & forecasting completed.")
    return True


if __name__ == "__main__":
    train_and_forecast()
