"""
FINAL Ensemble Engine (FIXED)
Compatible with all corrected models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("outputs")


# ==========================================
# LOAD FORECAST FILES
# ==========================================

def load_predictions():

    files = {
        "SARIMA": "sarima_forecast.csv",
        "LSTM": "lstm_forecast.csv",
        "XGB": "xgb_forecast.csv",
        "ELASTIC": "elastic_forecast.csv",
    }

    dfs = {}

    for name, file in files.items():

        path = OUTPUT_DIR / file

        if not path.exists():
            print(f"⚠️ Missing {file}")
            continue

        df = pd.read_csv(path)

        df["Date"] = pd.to_datetime(df["Date"], format='ISO8601')

        df = df.sort_values("Date")

        dfs[name] = df

        print(f"✅ Loaded {name}")

    return dfs


# ==========================================
# ALIGN DATES
# ==========================================

def align_dates(dfs):

    start = max(df["Date"].min() for df in dfs.values())
    end = min(df["Date"].max() for df in dfs.values())

    aligned = {}

    for name, df in dfs.items():

        f = df[(df["Date"] >= start) & (df["Date"] <= end)]

        aligned[name] = f.reset_index(drop=True)

    return aligned


# ==========================================
# CREATE ENSEMBLE
# ==========================================

def create_ensemble(dfs):

    dfs = align_dates(dfs)

    base = None

    for name, df in dfs.items():

        gold_col = [c for c in df.columns if "Gold" in c][0]
        silver_col = [c for c in df.columns if "Silver" in c][0]

        temp = df[["Date", gold_col, silver_col]]

        temp = temp.rename(
            columns={
                gold_col: f"{name}_Gold",
                silver_col: f"{name}_Silver",
            }
        )

        if base is None:
            base = temp
        else:
            base = pd.merge(base, temp, on="Date")

    gold_cols = [c for c in base.columns if "_Gold" in c]
    silver_cols = [c for c in base.columns if "_Silver" in c]

    gold_matrix = base[gold_cols].values
    silver_matrix = base[silver_cols].values

    weights = np.ones(len(gold_cols)) / len(gold_cols)

    gold_avg = np.average(gold_matrix, axis=1, weights=weights)
    silver_avg = np.average(silver_matrix, axis=1, weights=weights)

    gold_std = np.std(gold_matrix, axis=1)
    silver_std = np.std(silver_matrix, axis=1)

    out = pd.DataFrame({

        "Date": base["Date"],

        "Gold_Ensemble": gold_avg,
        "Silver_Ensemble": silver_avg,

        "Gold_Lower": gold_avg - 1.96 * gold_std,
        "Gold_Upper": gold_avg + 1.96 * gold_std,

        "Silver_Lower": silver_avg - 1.96 * silver_std,
        "Silver_Upper": silver_avg + 1.96 * silver_std,
    })

    return out


# ==========================================
# PLOT
# ==========================================

def plot(df):

    plt.figure(figsize=(12, 6))

    plt.plot(df["Date"], df["Gold_Ensemble"], label="Gold")
    plt.plot(df["Date"], df["Silver_Ensemble"], label="Silver")

    plt.legend()
    plt.grid()

    plt.savefig(OUTPUT_DIR / "ensemble_plot.png")

    print("📊 Plot saved")


# ==========================================
# RUN
# ==========================================

def run_ensemble():

    dfs = load_predictions()

    if len(dfs) < 2:
        print("❌ Need at least 2 models")
        return

    df = create_ensemble(dfs)

    df.to_csv(
        OUTPUT_DIR / "ensemble_forecast.csv",
        index=False,
    )

    plot(df)

    print("✅ Ensemble done")


if __name__ == "__main__":
    run_ensemble()