from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sync_csv_to_sqlite import store_factor_impact

try:
    import shap
except ImportError:  # pragma: no cover - exercised only when shap is unavailable
    shap = None

OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")

OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_BUNDLES = {
    "Gold": MODELS_DIR / "xgb_gold_bundle.pkl",
    "Silver": MODELS_DIR / "xgb_silver_bundle.pkl",
}


def load_bundle(target: str) -> dict:
    bundle_path = MODEL_BUNDLES[target]
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing XGBoost bundle for {target}: {bundle_path}")
    return joblib.load(bundle_path)


def explain_with_shap(model, features: pd.DataFrame) -> np.ndarray:
    if shap is not None:
        explainer = shap.Explainer(model)
        explanation = explainer(features)
        values = np.asarray(explanation.values)
        return values[0]

    # Fallback uses XGBoost's SHAP-compatible contribution scores when `shap`
    # is not installed in the active environment.
    matrix = xgb.DMatrix(features)
    contributions = model.get_booster().predict(matrix, pred_contribs=True)
    return contributions[0][:-1]


def calculate_factor_impact(bundle: dict) -> pd.DataFrame:
    feature_names = bundle["feature_names"]
    latest_features = bundle["latest_features"][feature_names].copy()
    shap_values = explain_with_shap(bundle["model"], latest_features)

    absolute_impact = np.abs(np.asarray(shap_values, dtype="float64"))
    total_impact = absolute_impact.sum()
    if total_impact <= 0:
        impact_percent = np.repeat(100.0 / len(feature_names), len(feature_names))
    else:
        impact_percent = (absolute_impact / total_impact) * 100.0

    return pd.DataFrame(
        {
            "date": bundle["latest_date"],
            "target": bundle["target"],
            "feature_name": feature_names,
            "impact_percent": impact_percent,
        }
    ).sort_values("impact_percent", ascending=False)


def run_shap_analysis() -> pd.DataFrame:
    print("[*] Running factor impact analysis...")
    frames = []

    for target in MODEL_BUNDLES:
        bundle = load_bundle(target)
        frames.append(calculate_factor_impact(bundle))

    factor_impact_df = pd.concat(frames, ignore_index=True)
    factor_impact_df.to_csv(OUTPUT_DIR / "factor_impact.csv", index=False)
    store_factor_impact(factor_impact_df)

    print("[OK] Factor impact analysis saved")
    return factor_impact_df


def main() -> pd.DataFrame:
    return run_shap_analysis()


if __name__ == "__main__":
    main()
