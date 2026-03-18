from pathlib import Path
import sys

# Ensure project root is on sys.path so sibling packages (Models, Ensemble) can be imported
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import subprocess
from Ensemble import ensemble_engine as ensemble_mod


def _run_script(rel_path):
    script = project_root / rel_path
    if not script.exists():
        print(f"Script not found: {script}")
        return None
    try:
        subprocess.run(["python", str(script)], check=True)
        return str(script)
    except subprocess.CalledProcessError as e:
        print(f"Script {script} failed: {e}")
        return None

def daily_run():
    print("Starting daily retraining pipeline...")

    sarima_path = _run_script(Path("Models") / "gold_rate_sarigma_model.py")
    xgb_path = _run_script(Path("Models") / "gold_silver_xgboost_model.py")
    lstm_path = _run_script(Path("Models") / "gold_silver_lstm_model.py")
    elastic_path = _run_script(Path("Models") / "gold_silver_elasticnet_model.py")

    try:
        ensemble_mod.run_ensemble(
            sarima_path,
            xgb_path,
            lstm_path,
            elastic_path,
        )
    except Exception as e:
        print(f"Warning: ensemble run failed: {e}")

    print("Daily pipeline completed successfully")

if __name__ == "__main__":
    daily_run()
