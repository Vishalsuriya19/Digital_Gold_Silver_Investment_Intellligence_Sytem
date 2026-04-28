from __future__ import annotations

from pathlib import Path

import pandas as pd

from Models.gold_rate_sarigma_model import train_and_forecast as train_sarimax
from Models.gold_silver_elasticnet_model import main as train_elasticnet
from Models.gold_silver_lstm_model import main as train_lstm
from Models.gold_silver_xgboost_model import train_and_forecast as train_xgboost
from data_fetcher import FACTOR_COLUMNS

DATA_FILE = Path("gold_silver_data.csv")


def validate_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    required_columns = {"Date", "Gold", "Silver"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    available_factors = [column for column in FACTOR_COLUMNS if column in df.columns]
    print(f"Available factor columns: {', '.join(available_factors) if available_factors else 'none'}")
    return df


def train_all_models() -> dict[str, bool]:
    validate_dataset()

    training_steps = [
        ("SARIMAX", train_sarimax, True),
        ("LSTM", train_lstm, False),
        ("XGBoost", train_xgboost, True),
        ("ElasticNet", train_elasticnet, False),
    ]

    results: dict[str, bool] = {}
    critical_failures: list[str] = []

    for name, trainer, critical in training_steps:
        print(f"\n=== TRAINING {name} ===")
        try:
            trainer()
            results[name] = True
            print(f"[OK] {name} completed")
        except Exception as exc:
            results[name] = False
            print(f"[ERROR] {name} failed: {exc}")
            if critical:
                critical_failures.append(name)

    successful_models = sum(results.values())
    if critical_failures:
        raise RuntimeError(f"Critical models failed: {', '.join(critical_failures)}")

    if successful_models < 2:
        raise RuntimeError("Need at least two successful models for ensemble generation.")

    return results


def main() -> dict[str, bool]:
    return train_all_models()


if __name__ == "__main__":
    main()
