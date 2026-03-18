# =====================================
# ElasticNet Gold Silver Forecast (FIXED)
# Uses local INR dataset
# =====================================

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path("gold_silver_data.csv")
OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")

FORECAST_DAYS = 30
LAGS = 30

OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# -------------------------
# Load dataset
# -------------------------
def load_dataset():

    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"], format='ISO8601')

    df = df.sort_values("Date")

    df.set_index("Date", inplace=True)

    return df


# -------------------------
# Create lag features
# -------------------------
def create_features(df):

    data = df.copy()

    for i in range(1, LAGS + 1):

        data[f"Gold_lag_{i}"] = data["Gold"].shift(i)
        data[f"Silver_lag_{i}"] = data["Silver"].shift(i)

    data = data.dropna()

    return data


# -------------------------
# Predict future
# -------------------------
def predict_future(model_g, model_s, last_row, scaler):

    row = last_row.copy()

    gold_preds = []
    silver_preds = []

    for _ in range(FORECAST_DAYS):

        X = scaler.transform(row.reshape(1, -1))

        g = model_g.predict(X)[0]
        s = model_s.predict(X)[0]

        gold_preds.append(g)
        silver_preds.append(s)

        row = np.roll(row, -2)
        row[-2] = g
        row[-1] = s

    return gold_preds, silver_preds


# -------------------------
# MAIN
# -------------------------
def main():

    print("[*] Loading dataset...")

    df = load_dataset()

    feat = create_features(df)

    X = feat.drop(["Gold", "Silver"], axis=1)

    y_gold = feat["Gold"]
    y_silver = feat["Silver"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X.values)

    print("[*] Training ElasticNet...")

    gold_model = ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=10000,
        tol=1e-4,
        random_state=42,
    )

    silver_model = ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=10000,
        tol=1e-4,
        random_state=42,
    )

    gold_model.fit(X_scaled, y_gold)
    silver_model.fit(X_scaled, y_silver)

    last_row = X.values[-1]  # use numpy array to avoid feature-name warnings

    g_pred, s_pred = predict_future(
        gold_model,
        silver_model,
        last_row,
        scaler,
    )

    future_dates = pd.date_range(
        start=df.index[-1] + timedelta(days=1),
        periods=FORECAST_DAYS,
        freq="D",
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Gold_ELASTIC": g_pred,
        "Silver_ELASTIC": s_pred,
    })

    forecast_df.to_csv(
        OUTPUT_DIR / "elastic_forecast.csv",
        index=False,
    )

    print("[OK] ElasticNet forecast saved")


if __name__ == "__main__":
    main()