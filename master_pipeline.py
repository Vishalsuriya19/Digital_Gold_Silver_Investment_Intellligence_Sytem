# =====================================
# GOLD & SILVER AI - MASTER PIPELINE
# FULL VERSION WITH AUTO ENGINE
# =====================================

import sys
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent


# =====================================
# RUN SCRIPT FUNCTION
# =====================================

def run_script(script_path, name):

    print("\n" + "=" * 80)
    print(f"RUNNING: {name}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        text=True
    )

    if result.returncode != 0:
        print(f"{name} FAILED")
        return False

    print(f"{name} COMPLETED")
    return True


# =====================================
# MAIN PIPELINE
# =====================================

def main():

    print("\nGOLD & SILVER AI AUTOMATED PIPELINE STARTED\n")

    # ==========================
    # STEP 1 - UPDATE DATASET
    # ==========================
    if not run_script(
        PROJECT_ROOT / "dataset_manager.py",
        "DATASET UPDATE"
    ):
        return


    # ==========================
    # STEP 2 - SARIMA TRAINING
    # ==========================
    if not run_script(
        PROJECT_ROOT / "model_trainer.py",
        "MODEL TRAINING (SARIMA)"
    ):
        return


    # ==========================
    # STEP 3 - OTHER MODELS
    # ==========================

    additional_models = [

        PROJECT_ROOT / "Models" / "gold_silver_lstm_model.py",
        PROJECT_ROOT / "Models" / "gold_silver_xgboost_model.py",
        PROJECT_ROOT / "Models" / "gold_silver_elasticnet_model.py",

    ]

    for script in additional_models:

        if not run_script(script, script.stem):
            print(f"WARNING: {script.stem} failed, continuing...")


    # ==========================
    # STEP 4 - ENSEMBLE
    # ==========================
    if not run_script(
        PROJECT_ROOT / "Ensemble" / "ensemble_engine.py",
        "ENSEMBLE ENGINE"
    ):
        return


    # ==========================
    # STEP 5 - RECOMMENDATION
    # ==========================
    if not run_script(
        PROJECT_ROOT / "recommendation_engine.py",
        "RECOMMENDATION ENGINE"
    ):
        return


    # ==========================
    # STEP 6 - AUTO INVESTMENT ENGINE
    # ==========================
    if not run_script(
        PROJECT_ROOT / "automation_engine" / "auto_engine.py",
        "AUTO INVESTMENT ENGINE"
    ):
        return


    print("\nFULL PIPELINE EXECUTED SUCCESSFULLY")


# =====================================
# RUN
# =====================================

if __name__ == "__main__":
    main()