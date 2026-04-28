"""
FINAL Ensemble Engine (FIXED)
Compatible with all corrected models
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("outputs")


def load_predictions():
    files = {
        "SARIMA": "sarima_forecast.csv",
        "LSTM": "lstm_forecast.csv",
        "XGB": "xgb_forecast.csv",
        "ELASTIC": "elastic_forecast.csv",
    }

    dfs = {}
    for name, file_name in files.items():
        path = OUTPUT_DIR / file_name
        if not path.exists():
            print(f"[WARN] Missing {file_name}")
            continue

        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        dfs[name] = df
        print(f"[OK] Loaded {name}")

    return dfs


def align_dates(dfs):
    start = max(df["Date"].min() for df in dfs.values())
    end = min(df["Date"].max() for df in dfs.values())

    aligned = {}
    for name, df in dfs.items():
        aligned[name] = df[(df["Date"] >= start) & (df["Date"] <= end)].reset_index(drop=True)

    return aligned


def create_ensemble(dfs):
    dfs = align_dates(dfs)

    base = None
    for name, df in dfs.items():
        gold_col = [column for column in df.columns if "Gold" in column][0]
        silver_col = [column for column in df.columns if "Silver" in column][0]

        temp = df[["Date", gold_col, silver_col]].rename(
            columns={
                gold_col: f"{name}_Gold",
                silver_col: f"{name}_Silver",
            }
        )

        if base is None:
            base = temp
        else:
            base = pd.merge(base, temp, on="Date")

    gold_cols = [column for column in base.columns if "_Gold" in column]
    silver_cols = [column for column in base.columns if "_Silver" in column]

    gold_matrix = base[gold_cols].values
    silver_matrix = base[silver_cols].values
    gold_avg, gold_std = robust_average_and_std(gold_matrix)
    silver_avg, silver_std = robust_average_and_std(silver_matrix)

    return pd.DataFrame(
        {
            "Date": base["Date"],
            "Gold_Ensemble": gold_avg,
            "Silver_Ensemble": silver_avg,
            "Gold_Lower": gold_avg - 1.96 * gold_std,
            "Gold_Upper": gold_avg + 1.96 * gold_std,
            "Silver_Lower": silver_avg - 1.96 * silver_std,
            "Silver_Upper": silver_avg + 1.96 * silver_std,
        }
    )


def robust_average_and_std(matrix):
    median = np.median(matrix, axis=1, keepdims=True)
    median_abs_deviation = np.median(np.abs(matrix - median), axis=1, keepdims=True)
    median_floor = np.maximum(np.abs(median) * 0.05, 1.0)
    spread = np.where(median_abs_deviation == 0, median_floor, median_abs_deviation)

    keep_mask = np.abs(matrix - median) <= (3.0 * spread)
    filtered = np.where(keep_mask, matrix, np.nan)

    average = np.nanmean(filtered, axis=1)
    std = np.nanstd(filtered, axis=1)

    average = np.where(np.isnan(average), median[:, 0], average)
    std = np.where(np.isnan(std), 0.0, std)
    return average, std


def plot(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Gold_Ensemble"], label="Gold")
    plt.plot(df["Date"], df["Silver_Ensemble"], label="Silver")
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT_DIR / "ensemble_plot.png")
    print("[OK] Plot saved")


def run_ensemble():
    dfs = load_predictions()

    if len(dfs) < 2:
        print("[ERROR] Need at least 2 models")
        return

    df = create_ensemble(dfs)
    df.to_csv(OUTPUT_DIR / "ensemble_forecast.csv", index=False)
    plot(df)
    print("[OK] Ensemble done")


if __name__ == "__main__":
    run_ensemble()
