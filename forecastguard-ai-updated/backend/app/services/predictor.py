import json
import sys
import os
from datetime import date as date_cls

import pandas as pd
import numpy as np
import joblib

from .feature_synth import STATE_META  # state -> (lat, lon, zone) lookup table only

# the ml/ package (explain.py, analogues.py, train.py's feature-name
# constants) lives at the project root, not inside backend/
_ML_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml")
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

# Resolve paths from the project root, not from the current working directory.
# This makes FastAPI work whether uvicorn is launched from the project root
# or from inside backend/.
_THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(_THIS_FILE_DIR, "..", "..", "..")
)

MODEL_DIR = os.path.abspath(
    os.environ.get(
        "FORECASTGUARD_MODEL_DIR",
        os.path.join(PROJECT_ROOT, "models")
    )
)

RAW_FORECAST_PATH = os.path.abspath(
    os.environ.get(
        "FORECASTGUARD_RAW_DATA",
        os.path.join(PROJECT_ROOT, "data", "raw", "real_forecast_obs.csv")
    )
)

NUMERIC_FEATURES = [
    "lead_day", "lat", "lon", "month", "day_of_year",
    "forecast_t2m", "forecast_msl", "forecast_rain", "forecast_u10", "forecast_v10",
    "forecast_rh", "forecast_z500", "forecast_u850", "forecast_v850", "forecast_q850",
    "forecast_wind_speed_10m", "forecast_wind_speed_850",
    "nbhd_rain_mean", "nbhd_rain_std", "nbhd_msl_std",
]
CATEGORICAL_FEATURES = ["region", "season"]


def risk_level(p):
    if p < 0.20:
        return "Very Low", "#34C9A6"
    if p < 0.40:
        return "Low", "#8BD17C"
    if p < 0.60:
        return "Moderate", "#F5C242"
    if p < 0.80:
        return "High", "#F2924B"
    return "Very High", "#E5484D"


def _window_mean_std(arr2d):
    """3x3 neighbourhood mean/std over a 2D grid (nearest-edge padding).
    Mirrors ml/build_features_and_labels.py so predict-time features match
    the features the model was trained on."""
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(arr2d, size=3, mode="nearest")
    mean_sq = uniform_filter(arr2d ** 2, size=3, mode="nearest")
    var = np.clip(mean_sq - mean ** 2, 0, None)
    std = np.sqrt(var)
    return mean, std


class ForecastGuardModel:
    """Singleton-style loader for all trained artifacts + the real HRES data."""
    _instance = None

    def __init__(self):
        self.preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.pkl")
        self.calibrated_model = joblib.load(f"{MODEL_DIR}/calibrated_model.pkl")

        with open(f"{MODEL_DIR}/metadata.json") as f:
            self.metadata = json.load(f)

        # Real HRES forecast + ERA5 verification pairs. This is the ONLY
        # source of "current forecast" numbers now - feature_synth.py's
        # synthesize_forecast() is no longer part of the live prediction
        # path, only its STATE_META lat/lon lookup table is still used.
        self.raw_df = pd.read_csv(RAW_FORECAST_PATH)
        self.raw_df["init_date"] = self.raw_df["init_date"].astype(str)
        self.available_dates = sorted(self.raw_df["init_date"].unique())

        # explainer / analogues are optional - degrade gracefully if the
        # extra artifact files haven't been generated yet
        self.explainer = None
        self.analogue_finder = None
        try:
            from explain import Explainer
            self.explainer = Explainer(MODEL_DIR)
        except Exception as e:
            print(f"[predictor] Explainer unavailable: {e}")
        try:
            from analogues import AnalogueFinder
            self.analogue_finder = AnalogueFinder(MODEL_DIR)
        except Exception as e:
            print(f"[predictor] AnalogueFinder unavailable: {e}")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _real_forecast_features(self, state: str, date_str: str, lead_day: int):
        """Real HRES forecast -> nearest grid cell -> feature vector, using
        the same feature engineering as ml/build_features_and_labels.py
        (wind speed from u/v components, 3x3 spatial-neighbourhood rainfall
        and pressure stats). This replaces the old synthesize_forecast()
        call: everything here comes from data/raw/real_forecast_obs.csv."""
        if state not in STATE_META:
            raise ValueError(f"Unknown state: {state}")
        target_lat, target_lon, _zone = STATE_META[state]

        subset = self.raw_df[
            (self.raw_df["init_date"] == date_str) & (self.raw_df["lead_day"] == lead_day)
        ]
        if subset.empty:
            raise ValueError(
                f"No real HRES forecast for init_date={date_str}, lead_day={lead_day}. "
                f"Available initialization dates in this prototype: {self.available_dates}"
            )

        dists = (subset["lat"] - target_lat) ** 2 + (subset["lon"] - target_lon) ** 2
        nearest = subset.loc[dists.idxmin()]

        pivot_rain = subset.pivot(index="lat", columns="lon", values="forecast_rain")
        pivot_msl = subset.pivot(index="lat", columns="lon", values="forecast_msl")
        rain_mean, rain_std = _window_mean_std(pivot_rain.values)
        _, msl_std = _window_mean_std(pivot_msl.values)
        lat_idx = list(pivot_rain.index).index(nearest["lat"])
        lon_idx = list(pivot_rain.columns).index(nearest["lon"])

        d = date_cls.fromisoformat(date_str)
        u10, v10 = float(nearest["forecast_u10"]), float(nearest["forecast_v10"])
        u850, v850 = float(nearest["forecast_u850"]), float(nearest["forecast_v850"])

        features = {
            "lead_day": lead_day, "lat": float(nearest["lat"]), "lon": float(nearest["lon"]),
            "month": d.month, "day_of_year": d.timetuple().tm_yday,
            "forecast_t2m": float(nearest["forecast_t2m"]), "forecast_msl": float(nearest["forecast_msl"]),
            "forecast_rain": float(nearest["forecast_rain"]), "forecast_u10": u10, "forecast_v10": v10,
            "forecast_rh": float(nearest["forecast_rh"]), "forecast_z500": float(nearest["forecast_z500"]),
            "forecast_u850": u850, "forecast_v850": v850, "forecast_q850": float(nearest["forecast_q850"]),
            "forecast_wind_speed_10m": float(np.sqrt(u10 ** 2 + v10 ** 2)),
            "forecast_wind_speed_850": float(np.sqrt(u850 ** 2 + v850 ** 2)),
            "nbhd_rain_mean": float(rain_mean[lat_idx, lon_idx]),
            "nbhd_rain_std": float(rain_std[lat_idx, lon_idx]),
            "nbhd_msl_std": float(msl_std[lat_idx, lon_idx]),
            "region": nearest["region"], "season": nearest["season"],
        }
        event_type = nearest["event_type"]
        return features, event_type

    def predict_for_state(self, state: str, date_str: str, lead_day: int):
        features, event_type = self._real_forecast_features(state, date_str, lead_day)
        row = pd.DataFrame([features])
        X = row[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        X_enc = self.preprocessor.transform(X)

        prob = float(self.calibrated_model.predict_proba(X_enc)[:, 1][0])
        label, color = risk_level(prob)

        result = {
            "state": state, "date": date_str, "lead_day": lead_day,
            "bust_probability": round(prob, 4),
            "reliability": round(1 - prob, 4),
            "risk_level": label, "risk_color": color,
            "season": features["season"], "region_zone": features["region"],
            "observed_event_type": event_type,
            "forecast_snapshot": {
                "rainfall_mm": round(features["forecast_rain"], 1),
                "temperature_c": round(features["forecast_t2m"], 1),
                "pressure_hpa": round(features["forecast_msl"], 1),
                "wind_speed_10m_kmh": round(features["forecast_wind_speed_10m"] * 3.6, 1),
            },
        }

        if self.explainer is not None:
            drivers, method = self.explainer.explain_instance(X_enc, top_k=5)
            result["drivers"] = drivers
            result["explanation_method"] = method
        else:
            result["drivers"] = []
            result["explanation_method"] = "unavailable"

        if self.analogue_finder is not None:
            result["historical_analogues"] = self.analogue_finder.find(X_enc, k=20)
        else:
            result["historical_analogues"] = None

        return result

    def predict_map(self, date_str: str, lead_day: int):
        cells = []
        for state in STATE_META.keys():
            p = self.predict_for_state(state, date_str, lead_day)
            cells.append({
                "state": state,
                "bust_probability": p["bust_probability"],
                "reliability": p["reliability"],
                "risk_level": p["risk_level"],
                "risk_color": p["risk_color"],
            })
        return {"date": date_str, "lead_day": lead_day, "cells": cells}
