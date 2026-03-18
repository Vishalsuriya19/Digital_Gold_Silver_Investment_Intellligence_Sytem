# ==========================================
# AI RECOMMENDATION ENGINE
# ==========================================

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("Outputs")


def generate_recommendation():
    file_path = OUTPUT_DIR / "ensemble_forecast.csv"

    if not file_path.exists():
        print("❌ Ensemble forecast not found.")
        return

    df = pd.read_csv(file_path)

    required = {"Gold_Ensemble", "Silver_Ensemble"}
    if not required.issubset(df.columns):
        print("❌ Ensemble output missing required columns.")
        return

    # Drop rows with NaNs in ensemble columns
    df = df.dropna(subset=["Gold_Ensemble", "Silver_Ensemble"]).reset_index(drop=True)

    if df.shape[0] < 2:
        print("❌ Not enough ensemble data to generate recommendation.")
        return

    gold_change = df["Gold_Ensemble"].iloc[-1] - df["Gold_Ensemble"].iloc[0]
    silver_change = df["Silver_Ensemble"].iloc[-1] - df["Silver_Ensemble"].iloc[0]

    def decision(change):
        if change > 1:
            return "BUY"
        elif change < -1:
            return "SELL"
        return "HOLD"

    recommendation = {
        "Gold": decision(gold_change),
        "Silver": decision(silver_change)
    }

    print("\nAI RECOMMENDATION")
    print("=================")
    print(f"Gold   → {recommendation['Gold']}")
    print(f"Silver → {recommendation['Silver']}")

    return recommendation


if __name__ == "__main__":
    generate_recommendation()
