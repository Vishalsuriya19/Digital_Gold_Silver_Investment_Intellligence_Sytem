from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_script(script_path: Path, name: str, required: bool = True) -> bool:
    print("\n" + "=" * 80)
    print(f"RUNNING: {name}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        text=True,
    )

    if result.returncode != 0:
        print(f"{name} FAILED")
        return not required

    print(f"{name} COMPLETED")
    return True


def main() -> None:
    print("\nGOLD & SILVER AI MULTIVARIATE PIPELINE STARTED\n")

    steps = [
        (PROJECT_ROOT / "dataset_manager.py", "FETCH + MERGE DATASET", True),
        (PROJECT_ROOT / "sync_csv_to_sqlite.py", "SYNC SQLITE MARKET DATA", True),
        (PROJECT_ROOT / "model_trainer.py", "TRAIN MULTIVARIATE MODELS", True),
        (PROJECT_ROOT / "shap_analysis.py", "RUN SHAP FACTOR ANALYSIS", True),
        (PROJECT_ROOT / "Ensemble" / "ensemble_engine.py", "GENERATE ENSEMBLE FORECAST", True),
        (PROJECT_ROOT / "recommendation_engine.py", "GENERATE RECOMMENDATION", False),
        (PROJECT_ROOT / "automation_engine" / "auto_engine.py", "RUN AUTO INVESTMENT ENGINE", False),
    ]

    for script_path, name, required in steps:
        if not run_script(script_path, name, required=required):
            return

    print("\nFULL PIPELINE EXECUTED SUCCESSFULLY")


if __name__ == "__main__":
    main()
