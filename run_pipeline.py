# =====================================
# GOLD & SILVER AI - MASTER PIPELINE
# FINAL FIXED VERSION
# =====================================

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


# =====================================
# RUN SCRIPT
# =====================================

def run_script(script_path, name):

    print("\n" + "=" * 80)
    print(f"RUNNING: {name}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(script_path)],
    )

    if result.returncode != 0:
        print(f"{name} FAILED")
        return False

    print(f"{name} DONE")
    return True


# =====================================
# MAIN PIPELINE
# =====================================

def main():

    print("\n🚀 GOLD & SILVER AI PIPELINE STARTED\n")

    # -------------------------
    # STEP 1 — DATASET
    # -------------------------

    if not run_script(
        PROJECT_ROOT / "dataset_manager.py",
        "DATASET UPDATE",
    ):
        return

    # -------------------------
    # STEP 2 — SARIMA
    # -------------------------

    if not run_script(
        PROJECT_ROOT / "model_trainer.py",
        "SARIMA MODEL",
    ):
        return

    # -------------------------
    # STEP 3 — LSTM / XGB / ELASTIC
    # -------------------------

    models = [

        PROJECT_ROOT / "Models" / "gold_silver_lstm_model.py",

        PROJECT_ROOT / "Models" / "gold_silver_xgboost_model.py",

        PROJECT_ROOT / "Models" / "gold_silver_elasticnet_model.py",

    ]

    for m in models:

        if not run_script(m, m.stem):
            print(f"⚠ {m.stem} failed, continuing")

    # -------------------------
    # STEP 4 — ENSEMBLE
    # -------------------------

    if not run_script(
        PROJECT_ROOT / "Ensemble" / "ensemble_engine.py",
        "ENSEMBLE",
    ):
        return

    # -------------------------
    # STEP 5 — RECOMMENDATION / AUTO BUY SELL
    # -------------------------

    if not run_script(
        PROJECT_ROOT / "recommendation_engine.py",
        "RECOMMENDATION",
    ):
        return

    print("\n✅ FULL PIPELINE COMPLETED")


if __name__ == "__main__":
    main()