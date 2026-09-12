"""
explain.py

Produces:
  1. GLOBAL feature importance (models/global_importance.json) - "what does
     the model rely on overall"
  2. A reusable Explainer class used by the backend to produce PER-PREDICTION
     explanations ("why is THIS forecast flagged as risky").

Uses SHAP (TreeExplainer) when the `shap` package is installed - this is the
preferred, standard approach for tree models and gives a proper additive
per-feature contribution for every single prediction.

If `shap` isn't installed in this environment, falls back to:
  - global importance: scikit-learn permutation_importance
  - per-prediction explanation: each feature's contribution is approximated
    by how many standard deviations it sits from the TRAINING mean,
    weighted by that feature's global importance. This is a reasonable,
    honestly-labelled stand-in - NOT a substitute for real SHAP values.
    Swap to SHAP the moment the package is available (pip install shap).

Run standalone to generate the global importance file:
    python ml/explain.py
"""

import json
import numpy as np
import pandas as pd
import joblib

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

MODEL_DIR = "models"

# Human-readable meteorological translations for each raw feature name.
FEATURE_EXPLANATIONS = {
    "lead_day": "Longer forecast lead time",
    "forecast_t2m": "Forecast near-surface temperature",
    "forecast_msl": "Forecast mean sea-level pressure",
    "forecast_rain": "Forecast rainfall intensity",
    "forecast_u10": "Forecast east-west surface wind component",
    "forecast_v10": "Forecast north-south surface wind component",
    "forecast_rh": "Forecast relative humidity",
    "forecast_z500": "Forecast 500 hPa geopotential height (upper-air pattern)",
    "forecast_u850": "Forecast 850 hPa wind (east-west)",
    "forecast_v850": "Forecast 850 hPa wind (north-south)",
    "forecast_q850": "Forecast 850 hPa moisture (lower-tropospheric humidity)",
    "forecast_wind_speed_10m": "Forecast near-surface wind speed",
    "forecast_wind_speed_850": "Forecast 850 hPa wind speed",
    "nbhd_rain_mean": "Neighbouring-region average rainfall forecast",
    "nbhd_rain_std": "Local spatial variability in the rainfall forecast",
    "nbhd_msl_std": "Local spatial variability in the pressure forecast",
    "month": "Time of year",
    "day_of_year": "Time of year",
    "lat": "Latitude",
    "lon": "Longitude",
    "region_north": "Northern-India regional pattern",
    "region_south": "Southern-India regional pattern",
    "region_east": "Eastern-India regional pattern",
    "region_west": "Western-India regional pattern",
    "region_central": "Central-India regional pattern",
    "region_northeast": "Northeast-India regional pattern",
    "season_monsoon": "Monsoon-season atmospheric regime",
    "season_winter": "Winter atmospheric regime",
    "season_pre_monsoon": "Pre-monsoon atmospheric regime",
    "season_post_monsoon": "Post-monsoon atmospheric regime",
}


def humanize(feature_name):
    return FEATURE_EXPLANATIONS.get(feature_name, feature_name.replace("_", " "))


class Explainer:
    def __init__(self, model_dir=MODEL_DIR):
        self.preprocessor = joblib.load(f"{model_dir}/preprocessor.pkl")
        self.raw_model = joblib.load(f"{model_dir}/raw_model.pkl")
        with open(f"{model_dir}/feature_names.json") as f:
            self.feature_names = json.load(f)

        self.train_means = None
        self.train_stds = None
        stats_path = f"{model_dir}/train_feature_stats.json"
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            self.train_means = np.array(stats["mean"])
            self.train_stds = np.array(stats["std"])
        except FileNotFoundError:
            pass

        if HAS_SHAP:
            try:
                self.shap_explainer = shap.TreeExplainer(self.raw_model)
            except Exception:
                self.shap_explainer = None
        else:
            self.shap_explainer = None

        with open(f"{model_dir}/global_importance.json") as f:
            self.global_importance = json.load(f)

    def explain_instance(self, X_row_encoded, top_k=5):
        """X_row_encoded: 1-row 2D array already run through preprocessor."""
        if self.shap_explainer is not None:
            shap_values = self.shap_explainer.shap_values(X_row_encoded)
            values = np.array(shap_values).reshape(-1)
            method = "shap"
        else:
            # fallback: z-score deviation from training mean, weighted by
            # global permutation importance
            z = (X_row_encoded.reshape(-1) - self.train_means) / (self.train_stds + 1e-9)
            weights = np.array([self.global_importance.get(f, 0.0) for f in self.feature_names])
            values = z * weights
            method = "zscore_weighted_fallback"

        order = np.argsort(-np.abs(values))[:top_k]
        drivers = []
        for i in order:
            fname = self.feature_names[i]
            drivers.append({
                "feature": fname,
                "label": humanize(fname),
                "contribution": float(values[i]),
                "direction": "increases_risk" if values[i] > 0 else "decreases_risk",
            })
        return drivers, method


def compute_and_save_global_importance():
    from sklearn.inspection import permutation_importance

    preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.pkl")
    model = joblib.load(f"{MODEL_DIR}/raw_model.pkl")

    df = pd.read_csv("data/processed/training_table.csv")

    from train import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, get_split_dates
    train_dates, val_dates, test_dates = get_split_dates(df)
    date_series = pd.to_datetime(df["init_date"]).dt.date
    test_mask = date_series.isin(test_dates)
    test_df = df[test_mask]
    if len(test_df) > 8000:
        test_df = test_df.sample(n=8000, random_state=42)

    X = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = test_df[TARGET]
    X_enc = preprocessor.transform(X)

    feature_names = (
        NUMERIC_FEATURES +
        list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    )
    with open(f"{MODEL_DIR}/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # save train feature stats for the z-score fallback explainer
    train_df = df[date_series.isin(train_dates)]
    X_train_enc = preprocessor.transform(train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    X_train_dense = np.asarray(X_train_enc.todense()) if hasattr(X_train_enc, "todense") else np.asarray(X_train_enc)
    with open(f"{MODEL_DIR}/train_feature_stats.json", "w") as f:
        json.dump({
            "mean": X_train_dense.mean(axis=0).tolist(),
            "std": X_train_dense.std(axis=0).tolist(),
        }, f)

    print("Computing permutation importance (global)...")
    result = permutation_importance(model, X_enc, y, n_repeats=5, random_state=42, scoring="average_precision")
    importance = dict(zip(feature_names, result.importances_mean.tolist()))
    # normalise to [0, 1] for readability
    max_v = max(max(importance.values()), 1e-9)
    importance = {k: max(v, 0) / max_v for k, v in importance.items()}

    with open(f"{MODEL_DIR}/global_importance.json", "w") as f:
        json.dump(importance, f, indent=2)

    top = sorted(importance.items(), key=lambda kv: -kv[1])[:10]
    print("Top global drivers of forecast-bust risk:")
    for fname, score in top:
        print(f"  {humanize(fname):45s} {score:.3f}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "ml")
    compute_and_save_global_importance()
    print(f"\nSHAP available: {HAS_SHAP}")
