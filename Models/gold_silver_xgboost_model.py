from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

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


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [
        column for column in FACTOR_COLUMNS
        if column in df.columns and df[column].notna().any()
    ]
    if not feature_columns:
        raise ValueError("No multivariate factor columns available for XGBoost training.")
    return feature_columns


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = get_feature_columns(df)
    training_df = df[["Gold", "Silver", *feature_columns]].copy()
    training_df = training_df.dropna()
    if training_df.empty:
        raise ValueError("Training dataset is empty after dropping missing rows.")
    return training_df, feature_columns


def create_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )


def build_importance_frame(
    model: XGBRegressor,
    feature_names: list[str],
    target: str,
    impact_date: str,
) -> pd.DataFrame:
    raw_importance = np.asarray(model.feature_importances_, dtype="float64")
    total = raw_importance.sum()
    if total <= 0:
        impact_percent = np.repeat(100.0 / len(feature_names), len(feature_names))
    else:
        impact_percent = (raw_importance / total) * 100.0

    return pd.DataFrame(
        {
            "date": impact_date,
            "target": target,
            "feature_name": feature_names,
            "impact_percent": impact_percent,
        }
    ).sort_values("impact_percent", ascending=False)


def save_model_bundle(
    model: XGBRegressor,
    target: str,
    feature_names: list[str],
    training_features: pd.DataFrame,
    future_features: pd.DataFrame,
    latest_date: str,
) -> None:
    bundle = {
        "model": model,
        "target": target,
        "feature_names": feature_names,
        "training_features": training_features.tail(min(180, len(training_features))).copy(),
        "latest_features": training_features.tail(1).copy(),
        "future_features": future_features.copy(),
        "latest_date": latest_date,
    }
    joblib.dump(bundle, MODELS_DIR / f"xgb_{target.lower()}_bundle.pkl")
    joblib.dump(model, MODELS_DIR / f"xgb_{target.lower()}.pkl")


def train_and_forecast(forecast_days: int = FORECAST_DAYS) -> pd.DataFrame:
    print("[*] Loading dataset for XGBoost...")
    df = load_dataset()
    training_df, feature_columns = build_training_frame(df)
    X = training_df[feature_columns].copy()

    print("[*] Training multivariate XGBoost models...")
    gold_model = create_model()
    silver_model = create_model()

    gold_model.fit(X, training_df["Gold"])
    silver_model.fit(X, training_df["Silver"])

    future_features = build_future_factor_frame(X, forecast_days)
    gold_forecast = gold_model.predict(future_features[feature_columns])
    silver_forecast = silver_model.predict(future_features[feature_columns])

    forecast_df = pd.DataFrame(
        {
            "Date": future_features.index,
            "Gold_XGB": gold_forecast,
            "Silver_XGB": silver_forecast,
        }
    )
    forecast_df.to_csv(OUTPUT_DIR / "xgb_forecast.csv", index=False)

    latest_date = training_df.index[-1].strftime("%Y-%m-%d")
    gold_importance = build_importance_frame(gold_model, feature_columns, "Gold", latest_date)
    silver_importance = build_importance_frame(silver_model, feature_columns, "Silver", latest_date)
    importance_df = pd.concat([gold_importance, silver_importance], ignore_index=True)
    importance_df.to_csv(OUTPUT_DIR / "xgb_feature_importance.csv", index=False)

    save_model_bundle(
        model=gold_model,
        target="Gold",
        feature_names=feature_columns,
        training_features=X,
        future_features=future_features,
        latest_date=latest_date,
    )
    save_model_bundle(
        model=silver_model,
        target="Silver",
        feature_names=feature_columns,
        training_features=X,
        future_features=future_features,
        latest_date=latest_date,
    )

    print("[OK] XGBoost forecast saved")
    return forecast_df


def main() -> pd.DataFrame:
    return train_and_forecast()


if __name__ == "__main__":
    main()
