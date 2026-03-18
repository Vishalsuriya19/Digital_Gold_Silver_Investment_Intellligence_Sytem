# -*- coding: utf-8 -*-
"""
Auto-fixer script for all model files
"""

from pathlib import Path

# Define the fixes for each model file
MODELS_DIR = Path(__file__).parent / "Models"

# SARIMA fix
sarima_code = '''# =====================================
# SARIMA Gold & Silver Forecast Pipeline
# =====================================

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path

# Config
START_DATE = "2015-01-01"
FORECAST_DAYS = 30
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TODAY = datetime.date.today().strftime("%Y-%m-%d")

def load_market_data(ticker):
    df = yf.download(ticker, start=START_DATE, end=TODAY, progress=False)
    df = df[['Close']]
    df.dropna(inplace=True)
    df = df.asfreq('B', method='ffill')
    return df

def train_sarima(series):
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)

def forecast_30_days(model):
    forecast = model.get_forecast(steps=FORECAST_DAYS)
    return forecast.predicted_mean, forecast.conf_int()

def plot_forecast(df, mean, ci, title, color):
    plt.figure(figsize=(12, 5))
    plt.plot(df['Close'], label="Historical Price")
    plt.plot(mean, label="30-Day Forecast", color=color)
    plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.3)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.close()

def main():
    print("[*] Fetching fresh market data...")
    gold_df = load_market_data("GC=F")
    silver_df = load_market_data("SI=F")

    print("[*] Training SARIMA models...")
    gold_model = train_sarima(gold_df['Close'])
    silver_model = train_sarima(silver_df['Close'])

    joblib.dump(gold_model, MODELS_DIR / "sarima_gold_model.pkl")
    joblib.dump(silver_model, MODELS_DIR / "sarima_silver_model.pkl")

    print("[*] Generating 30-day forecasts...")
    gold_mean, gold_ci = forecast_30_days(gold_model)
    silver_mean, silver_ci = forecast_30_days(silver_model)

    print("\\n[GOLD] Next 30 Days")
    print(gold_mean)

    print("\\n[SILVER] Next 30 Days")
    print(silver_mean)

    predictions_df = pd.DataFrame({
        'Date': gold_mean.index,
        'Gold_Predicted': gold_mean.values,
        'Silver_Predicted': silver_mean.values
    })
    
    output_dir = Path(__file__).parent.parent / "Outputs"
    output_dir.mkdir(exist_ok=True)
    predictions_df.to_csv(output_dir / "sarima_predictions.csv", index=False)

    plot_forecast(gold_df, gold_mean, gold_ci, "Gold Price - 30 Day SARIMA Forecast", "red")
    plot_forecast(silver_df, silver_mean, silver_ci, "Silver Price - 30 Day SARIMA Forecast", "green")

    print("\\n[OK] SARIMA model completed. Predictions saved.")

if __name__ == "__main__":
    main()
'''

# LSTM fix
lstm_code = '''# -*- coding: utf-8 -*-
"""Gold_Silver_LSTM_model"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
from pathlib import Path
import joblib

warnings.filterwarnings("ignore")

def main():
    start_date = "2015-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    print("[*] Downloading gold and silver data...")
    gold = yf.download("GC=F", start=start_date, end=end_date, progress=False)
    silver = yf.download("SI=F", start=start_date, end=end_date, progress=False)

    # Handle MultiIndex columns from yfinance
    if isinstance(gold.columns, pd.MultiIndex):
        gold = gold['Close']
    if isinstance(silver.columns, pd.MultiIndex):
        silver = silver['Close']

    data = pd.DataFrame({
        'Gold': gold,
        'Silver': silver
    }).dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for LSTM
    def create_sequences(dataset, lookback=60):
        X, y = [], []
        for i in range(lookback, len(dataset)):
            X.append(dataset[i-lookback:i])
            y.append(dataset[i])
        return np.array(X), np.array(y)

    LOOKBACK = 60
    X, y = create_sequences(scaled_data, LOOKBACK)

    # Build and train LSTM model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(50),
        Dense(2)  # Gold & Silver
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    print("[*] Training LSTM model...")
    early_stop = EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X, y,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Generate predictions
    def predict_next_30_days_lstm(model, last_sequence, scaler, lookback=60):
        future_predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(30):
            pred = model.predict(current_sequence.reshape(1, lookback, 2), verbose=0)
            future_predictions.append(pred[0])
            current_sequence = np.vstack([current_sequence[1:], pred])

        future_predictions = scaler.inverse_transform(future_predictions)
        return future_predictions

    last_sequence = scaled_data[-LOOKBACK:]
    future_prices = predict_next_30_days_lstm(model, last_sequence, scaler)

    dates = [datetime.today() + timedelta(days=i) for i in range(1, 31)]

    forecast_df = pd.DataFrame({
        'Date': dates,
        'Gold_Predicted': future_prices[:, 0],
        'Silver_Predicted': future_prices[:, 1]
    })

    # Save predictions to CSV
    output_dir = Path(__file__).parent.parent / "Outputs"
    output_dir.mkdir(exist_ok=True)
    forecast_df.to_csv(output_dir / "lstm_predictions.csv", index=False)

    print("[OK] LSTM model completed. Predictions saved.")
    print(forecast_df.head())

if __name__ == "__main__":
    main()
'''

# XGBoost fix
xgboost_code = '''# -*- coding: utf-8 -*-
"""Gold & Silver XGBoost Model for 30-Day Forecasting"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

def main():
    start_date = "2015-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    print("[*] Fetching data...")
    gold = yf.download("GC=F", start=start_date, end=end_date, progress=False)
    silver = yf.download("SI=F", start=start_date, end=end_date, progress=False)

    # Handle MultiIndex columns
    if isinstance(gold.columns, pd.MultiIndex):
        gold = gold['Close']
    if isinstance(silver.columns, pd.MultiIndex):
        silver = silver['Close']

    # Ensure they are Series and reset index
    gold = pd.Series(gold.values, index=gold.index)
    silver = pd.Series(silver.values, index=silver.index)

    # Merge data
    data = pd.DataFrame({
        'Gold_Close': gold,
        'Silver_Close': silver
    }).dropna()

    # Create features
    def create_features(df, lags=30):
        df_feat = df.copy()
        for lag in range(1, lags + 1):
            df_feat[f'Gold_lag_{lag}'] = df_feat['Gold_Close'].shift(lag)
            df_feat[f'Silver_lag_{lag}'] = df_feat['Silver_Close'].shift(lag)
        return df_feat.dropna()

    data_feat = create_features(data, lags=30)

    # Prepare training data
    X = data_feat.drop(['Gold_Close', 'Silver_Close'], axis=1)
    y_gold = data_feat['Gold_Close']
    y_silver = data_feat['Silver_Close']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train models
    print("[*] Training XGBoost models...")
    gold_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    silver_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    gold_model.fit(X_scaled, y_gold)
    silver_model.fit(X_scaled, y_silver)

    # Prediction function
    def predict_next_30_days(model_gold, model_silver, last_data, scaler):
        future_gold = []
        future_silver = []
        current_data = last_data.copy()

        for _ in range(30):
            X_input = scaler.transform(current_data.values.reshape(1, -1))
            gold_pred = model_gold.predict(X_input)[0]
            silver_pred = model_silver.predict(X_input)[0]

            future_gold.append(gold_pred)
            future_silver.append(silver_pred)

            # Shift lags
            current_data = pd.Series(
                np.roll(current_data.values, -2),
                index=current_data.index
            )
            current_data[-2] = gold_pred
            current_data[-1] = silver_pred

        return future_gold, future_silver

    # Generate predictions
    last_row = data_feat.drop(['Gold_Close', 'Silver_Close'], axis=1).iloc[-1]
    gold_30, silver_30 = predict_next_30_days(gold_model, silver_model, last_row, scaler)

    dates = [datetime.today() + timedelta(days=i) for i in range(1, 31)]
    forecast_df = pd.DataFrame({
        'Date': dates,
        'Gold_Predicted': gold_30,
        'Silver_Predicted': silver_30
    })

    # Save predictions
    output_dir = Path(__file__).parent.parent / "Outputs"
    output_dir.mkdir(exist_ok=True)
    forecast_df.to_csv(output_dir / "xgboost_predictions.csv", index=False)

    print("[OK] XGBoost model completed. Predictions saved.")
    print(forecast_df.head())

if __name__ == "__main__":
    main()
'''

# ElasticNet fix
elasticnet_code = '''# -*- coding: utf-8 -*-
"""Gold_Silver_Elasticnet_model"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

def main():
    start_date = "2015-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    print("[*] Downloading data...")
    gold = yf.download("GC=F", start=start_date, end=end_date, progress=False)
    silver = yf.download("SI=F", start=start_date, end=end_date, progress=False)

    # Handle MultiIndex columns from yfinance
    if isinstance(gold.columns, pd.MultiIndex):
        gold = gold['Close']
    if isinstance(silver.columns, pd.MultiIndex):
        silver = silver['Close']

    # Ensure they are Series and reset index
    gold = pd.Series(gold.values, index=gold.index)
    silver = pd.Series(silver.values, index=silver.index)

    data = pd.DataFrame({
        'Gold': gold,
        'Silver': silver
    }).dropna()

    def create_lag_features(df, lags=30):
        df_lag = df.copy()
        for lag in range(1, lags + 1):
            df_lag[f'Gold_lag_{lag}'] = df_lag['Gold'].shift(lag)
            df_lag[f'Silver_lag_{lag}'] = df_lag['Silver'].shift(lag)
        return df_lag.dropna()

    LAGS = 30
    data_lag = create_lag_features(data, LAGS)

    X = data_lag.drop(['Gold', 'Silver'], axis=1)
    y_gold = data_lag['Gold']
    y_silver = data_lag['Silver']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[*] Training ElasticNet models...")
    gold_model = ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=5000,
        random_state=42
    )

    silver_model = ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=5000,
        random_state=42
    )

    gold_model.fit(X_scaled, y_gold)
    silver_model.fit(X_scaled, y_silver)

    def predict_next_30_days_elasticnet(
        gold_model, silver_model, last_row, scaler, lags=30
    ):
        gold_preds = []
        silver_preds = []

        current_features = last_row.copy()

        for _ in range(30):
            X_input = scaler.transform(current_features.values.reshape(1, -1))

            gold_pred = gold_model.predict(X_input)[0]
            silver_pred = silver_model.predict(X_input)[0]

            gold_preds.append(gold_pred)
            silver_preds.append(silver_pred)

            # Shift lag features
            current_features = pd.Series(
                np.roll(current_features.values, -2),
                index=current_features.index
            )
            current_features[-2] = gold_pred
            current_features[-1] = silver_pred

        return gold_preds, silver_preds

    last_features = data_lag.drop(['Gold', 'Silver'], axis=1).iloc[-1]

    gold_30, silver_30 = predict_next_30_days_elasticnet(
        gold_model,
        silver_model,
        last_features,
        scaler,
        LAGS
    )

    dates = [datetime.today() + timedelta(days=i) for i in range(1, 31)]

    forecast_df = pd.DataFrame({
        'Date': dates,
        'Gold_Predicted': gold_30,
        'Silver_Predicted': silver_30
    })

    # Save predictions to CSV
    output_dir = Path(__file__).parent.parent / "Outputs"
    output_dir.mkdir(exist_ok=True)
    forecast_df.to_csv(output_dir / "elasticnet_predictions.csv", index=False)

    print("[OK] ElasticNet model completed. Predictions saved.")
    print(forecast_df.head())

if __name__ == "__main__":
    main()
'''

# Write all files
def apply_fixes():
    files = {
        MODELS_DIR / "gold_rate_sarigma_model.py": sarima_code,
        MODELS_DIR / "gold_silver_lstm_model.py": lstm_code,
        MODELS_DIR / "gold_silver_xgboost_model.py": xgboost_code,
        MODELS_DIR / "gold_silver_elasticnet_model.py": elasticnet_code,
    }
    
    for filepath, code in files.items():
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"[OK] Fixed: {filepath.name}")

if __name__ == "__main__":
    print("[*] Applying fixes to all model files...")
    apply_fixes()
    print("[OK] All files updated successfully!")