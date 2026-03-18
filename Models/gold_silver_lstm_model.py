# =====================================
# LSTM / MLP Gold Silver Forecast (FIXED)
# Uses local INR dataset
# =====================================

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

DATA_FILE = Path("gold_silver_data.csv")
OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")

FORECAST_DAYS = 30
LOOKBACK = 60

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
# Create sequences
# -------------------------
def create_sequences(data, lookback):

    X = []
    y = []

    for i in range(lookback, len(data)):

        X.append(data[i - lookback:i])
        y.append(data[i])

    return np.array(X), np.array(y)


# -------------------------
# Predict future
# -------------------------
def predict_future(model, last_seq, scaler):

    seq = last_seq.copy()

    preds = []

    for _ in range(FORECAST_DAYS):

        x = seq.reshape(1, -1)

        pred = model.predict(x)[0]

        preds.append(pred)

        seq = np.vstack([seq[1:], pred])

    preds = scaler.inverse_transform(preds)

    return preds


# -------------------------
# MAIN
# -------------------------
def main():

    print("[*] Loading dataset...")

    df = load_dataset()

    data = df[["Gold", "Silver"]].values

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(data)

    X, y = create_sequences(scaled, LOOKBACK)

    X = X.reshape(X.shape[0], -1)

    print("[*] Training LSTM/MLP...")

    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42,
    )

    model.fit(X, y)

    last_seq = scaled[-LOOKBACK:]

    future = predict_future(model, last_seq, scaler)

    future_dates = pd.date_range(
        start=df.index[-1] + timedelta(days=1),
        periods=FORECAST_DAYS,
        freq="D",
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Gold_LSTM": future[:, 0],
        "Silver_LSTM": future[:, 1],
    })

    forecast_df.to_csv(OUTPUT_DIR / "lstm_forecast.csv", index=False)

    print("[OK] LSTM forecast saved")


if __name__ == "__main__":
    main()