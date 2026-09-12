"""
train.py

Loads data/processed/training_table.csv, does a TEMPORAL split (never a
random split - weather is autocorrelated in time), trains:
  1. A logistic regression baseline
  2. A gradient-boosted tree model (XGBoost if installed, otherwise
     scikit-learn's HistGradientBoostingClassifier as a drop-in fallback
     with a very similar algorithm - swap back to XGBoost/LightGBM the
     moment you have internet access in your own environment; the rest
     of the pipeline does not need to change)

then reports PR-AUC / ROC-AUC / Recall / Precision / Brier score for both,
split out by lead_day, so you can see how skill degrades with forecast
horizon (exactly what NCMRWF/SIH judges will want to see).

Run:
    python ml/train.py
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, brier_score_loss, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
import joblib

try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN_ESTIMATOR = True
except ImportError:
    HAS_FROZEN_ESTIMATOR = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import HistGradientBoostingClassifier

DATA_PATH = "data/processed/training_table.csv"
MODEL_DIR = "models"

NUMERIC_FEATURES = [
    "lead_day", "lat", "lon", "month", "day_of_year",
    "forecast_t2m", "forecast_msl", "forecast_rain", "forecast_u10", "forecast_v10",
    "forecast_rh", "forecast_z500", "forecast_u850", "forecast_v850", "forecast_q850",
    "forecast_wind_speed_10m", "forecast_wind_speed_850",
    "nbhd_rain_mean", "nbhd_rain_std", "nbhd_msl_std",
]
CATEGORICAL_FEATURES = ["region", "season"]
TARGET = "bust"

# Columns that must NEVER be used as model inputs (they are derived from the
# actual/verification data and would leak the label into the features)
LEAKAGE_COLS = ["abs_error_rain", "abs_error_t2m", "abs_error_msl", "event_type"]


def get_split_dates(df):
    """Chronological initialization-date boundaries for the prototype split.

    With only a handful of real HRES+ERA5 initialization dates available
    (no 2016-2022 archive downloaded yet), we can't split by *year* the way
    a multi-year production system would. Instead we split by *date*, using
    the earliest dates for training, the second-to-last date for
    calibration/validation, and the single latest date as a held-out test
    set. This preserves the one property that actually matters for a valid
    temporal evaluation: test data is strictly later in time than training
    data.
    """
    dates = sorted(pd.to_datetime(df["init_date"]).dt.date.unique())

    if len(dates) < 3:
        raise ValueError(
            "Need at least 3 initialization dates for the prototype "
            "train/validation/test split."
        )

    train_dates = dates[:-2]
    val_dates = [dates[-2]]
    test_dates = [dates[-1]]
    return train_dates, val_dates, test_dates


def temporal_split(df):
    train_dates, val_dates, test_dates = get_split_dates(df)
    date_series = pd.to_datetime(df["init_date"]).dt.date

    train = df[date_series.isin(train_dates)].copy()
    val = df[date_series.isin(val_dates)].copy()
    test = df[date_series.isin(test_dates)].copy()

    return train, val, test


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def evaluate(y_true, y_prob, label=""):
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = dict(
        roc_auc=roc_auc_score(y_true, y_prob),
        pr_auc=average_precision_score(y_true, y_prob),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        brier=brier_score_loss(y_true, y_prob),
    )
    print(f"[{label}] " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    return metrics


def evaluate_by_lead(df, y_prob, label=""):
    out = []
    d = df.copy()
    d["y_prob"] = y_prob
    for lead_day, g in d.groupby("lead_day"):
        if g[TARGET].nunique() < 2:
            continue
        m = evaluate(g[TARGET], g["y_prob"], label=f"{label} lead={lead_day}")
        m["lead_day"] = int(lead_day)
        out.append(m)
    return pd.DataFrame(out)


def main():
    print("Loading processed training table...")
    df = pd.read_csv(DATA_PATH)

    train_df, val_df, test_df = temporal_split(df)
    train_dates, val_dates, test_dates = get_split_dates(df)
    print(
        f"Train: {len(train_df):,} (init dates {train_dates[0]}..{train_dates[-1]})  "
        f"Val: {len(val_df):,} ({val_dates[0]})  "
        f"Test: {len(test_df):,} ({test_dates[0]})"
    )
    print(
        "NOTE: this is a prototype evaluation on a small 2019 sample "
        "(one initialization date held out for test), not a production "
        "multi-year backtest."
    )

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET]
    X_val = val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_val = val_df[TARGET]
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET]

    # ---------------- Baseline: Logistic Regression ----------------
    print("\n=== Training baseline: Logistic Regression ===")
    baseline = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    baseline.fit(X_train, y_train)
    baseline_test_prob = baseline.predict_proba(X_test)[:, 1]
    baseline_metrics = evaluate(y_test, baseline_test_prob, label="LogReg/TEST")

    # ---------------- Main model: XGBoost or HistGBM fallback ----------------
    model_name = "XGBoost" if HAS_XGBOOST else "HistGradientBoostingClassifier (XGBoost fallback)"
    print(f"\n=== Training main model: {model_name} ===")

    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)
    X_test_enc = preprocessor.transform(X_test)

    if HAS_XGBOOST:
        clf = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        )
        clf.fit(X_train_enc, y_train)
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=400, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=42,
        )
        clf.fit(X_train_enc, y_train)

    raw_test_prob = clf.predict_proba(X_test_enc)[:, 1]
    raw_metrics = evaluate(y_test, raw_test_prob, label=f"{model_name} raw/TEST")

    # ---------------- Calibration (on validation set, not test) ----------------
    print("\n=== Calibrating probabilities (isotonic, fit on 2023 validation set) ===")
    if HAS_FROZEN_ESTIMATOR:
        # scikit-learn >= 1.6 style: wrap the already-fitted model so
        # CalibratedClassifierCV treats it as fixed and only fits the
        # calibration map on the validation set.
        calibrated = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
        calibrated.fit(X_val_enc, y_val)
    else:
        # older scikit-learn style
        calibrated = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
        calibrated.fit(X_val_enc, y_val)
    calibrated_test_prob = calibrated.predict_proba(X_test_enc)[:, 1]
    calibrated_metrics = evaluate(y_test, calibrated_test_prob, label=f"{model_name} calibrated/TEST")

    print("\n=== Skill by forecast lead day (calibrated model, TEST) ===")
    by_lead = evaluate_by_lead(test_df, calibrated_test_prob, label="calibrated")

    # ---------------- Save everything ----------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.pkl")
    joblib.dump(clf, f"{MODEL_DIR}/raw_model.pkl")
    joblib.dump(calibrated, f"{MODEL_DIR}/calibrated_model.pkl")
    joblib.dump(baseline, f"{MODEL_DIR}/baseline_logreg.pkl")

    metadata = dict(
        model_name=model_name,
        used_xgboost=HAS_XGBOOST,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        target=TARGET,
        train_period="earlier prototype initialization dates",
        val_period="middle prototype initialization date (used for calibration)",
        test_period="latest prototype initialization date",
        evaluation_note="Prototype evaluation on a small 2019 sample — not production performance",
        train_dates=[str(d) for d in get_split_dates(df)[0]],
        val_dates=[str(d) for d in get_split_dates(df)[1]],
        test_dates=[str(d) for d in get_split_dates(df)[2]],
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
        baseline_metrics=baseline_metrics,
        raw_model_metrics=raw_metrics,
        calibrated_model_metrics=calibrated_metrics,
        by_lead_day=by_lead.to_dict(orient="records"),
    )
    with open(f"{MODEL_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=float)

    by_lead.to_csv(f"{MODEL_DIR}/metrics_by_lead_day.csv", index=False)

    cm = confusion_matrix(y_test, (calibrated_test_prob >= 0.5).astype(int))
    np.savetxt(f"{MODEL_DIR}/confusion_matrix_test.csv", cm, fmt="%d", delimiter=",")

    print("\nSaved model artifacts + metadata.json + metrics_by_lead_day.csv to models/")
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("by_lead_day",)}, indent=2, default=float))


if __name__ == "__main__":
    main()
