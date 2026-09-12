"""
build_features_and_labels.py

Takes data/raw/synthetic_forecast_obs.csv (forecast + verification pairs)
and produces data/processed/training_table.csv with:
  - forecast errors per variable
  - a scientifically-motivated BUST label (per lead_day x season x variable
    90th-percentile threshold, computed ONLY from a "climatology" period so
    later train/val/test splits don't leak)
  - engineered features that are available AT FORECAST TIME (never using
    the actual/verification columns as model inputs -> avoids leakage)
  - spatial neighbourhood (3x3 window) statistics per init_date/lead_day

Run:
    python ml/build_features_and_labels.py
"""

import numpy as np
import pandas as pd
import os

RAW_PATH = "data/raw/real_forecast_obs.csv"
OUT_PATH = "data/processed/training_table.csv"
THRESH_PATH = "data/processed/bust_thresholds.csv"

# Percentile used to define "unusually large" error -> a bust
BUST_PERCENTILE = 0.90

# The label is primarily driven by rainfall error (highest-impact variable),
# but a forecast also counts as a bust if temperature or pressure error is
# extreme, since those matter operationally too.
LABEL_VARS = ["rain", "t2m", "msl"]


def month_to_season(m):
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "pre_monsoon"
    if m in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"


def add_errors(df):
    for v in ["t2m", "msl", "rain", "u10", "v10", "rh", "z500", "u850", "v850", "q850"]:
        df[f"error_{v}"] = df[f"forecast_{v}"] - df[f"actual_{v}"]
        df[f"abs_error_{v}"] = df[f"error_{v}"].abs()
    return df


def compute_thresholds(train_df):
    """Compute BUST_PERCENTILE thresholds per (lead_day, season, variable)
    using ONLY the training period -> avoids leaking future information
    into the label definition itself."""
    rows = []
    for v in LABEL_VARS:
        g = train_df.groupby(["lead_day", "season"])[f"abs_error_{v}"].quantile(BUST_PERCENTILE)
        for (lead_day, season), val in g.items():
            rows.append(dict(lead_day=lead_day, season=season, variable=v, threshold=val))
    return pd.DataFrame(rows)


def apply_labels(df, thresholds):
    df = df.merge(
        thresholds.pivot(index=["lead_day", "season"], columns="variable", values="threshold")
        .rename(columns={v: f"thresh_{v}" for v in LABEL_VARS})
        .reset_index(),
        on=["lead_day", "season"], how="left"
    )
    bust_any = np.zeros(len(df), dtype=bool)
    for v in LABEL_VARS:
        bust_any |= (df[f"abs_error_{v}"] > df[f"thresh_{v}"])
    df["bust"] = bust_any.astype(int)
    return df


def add_derived_features(df):
    # wind speed (available at forecast time - built from forecast fields)
    df["forecast_wind_speed_10m"] = np.sqrt(df["forecast_u10"]**2 + df["forecast_v10"]**2)
    df["forecast_wind_speed_850"] = np.sqrt(df["forecast_u850"]**2 + df["forecast_v850"]**2)

    df["init_date_dt"] = pd.to_datetime(df["init_date"])
    df["month"] = df["init_date_dt"].dt.month
    df["day_of_year"] = df["init_date_dt"].dt.dayofyear
    df["year"] = df["init_date_dt"].dt.year
    return df


def _window_mean_std(arr2d):
    """3x3 neighbourhood mean/std over a 2D grid using uniform_filter
    (nearest-edge padding), returned as two same-shape arrays."""
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(arr2d, size=3, mode="nearest")
    mean_sq = uniform_filter(arr2d**2, size=3, mode="nearest")
    var = np.clip(mean_sq - mean**2, 0, None)
    std = np.sqrt(var)
    return mean, std


def add_spatial_neighbourhood_features(df):
    """3x3 neighbourhood mean/std of forecast rainfall & pressure per
    (init_date, lead_day) -> helps the model see spatially unstable
    structures, not just an isolated grid point."""
    df = df.sort_values(["init_date", "lead_day", "lat", "lon"]).reset_index(drop=True)

    pieces = []
    for (init_date, lead_day), group in df.groupby(["init_date", "lead_day"], sort=False):
        pivot_rain = group.pivot(index="lat", columns="lon", values="forecast_rain")
        pivot_msl = group.pivot(index="lat", columns="lon", values="forecast_msl")

        rain_mean, rain_std = _window_mean_std(pivot_rain.values)
        _, msl_std = _window_mean_std(pivot_msl.values)

        stats = pd.DataFrame({
            "lat": np.repeat(pivot_rain.index.values, pivot_rain.shape[1]),
            "lon": np.tile(pivot_rain.columns.values, pivot_rain.shape[0]),
            "nbhd_rain_mean": rain_mean.flatten(),
            "nbhd_rain_std": rain_std.flatten(),
            "nbhd_msl_std": msl_std.flatten(),
        })
        stats["init_date"] = init_date
        stats["lead_day"] = lead_day
        pieces.append(stats)

    nbhd = pd.concat(pieces, ignore_index=True)

    df = df.merge(nbhd, on=["init_date", "lead_day", "lat", "lon"], how="left")
    df["nbhd_rain_std"] = df["nbhd_rain_std"].fillna(0)
    df["nbhd_msl_std"] = df["nbhd_msl_std"].fillna(0)
    return df


def main():
    print("Loading raw data...")
    df = pd.read_csv(RAW_PATH)
    df = add_errors(df)
    df = add_derived_features(df)

    print("Adding spatial neighbourhood features (this takes a minute)...")
    df = add_spatial_neighbourhood_features(df)

    # Temporal split boundaries used ONLY to compute the bust threshold
    # without leakage (matches the train period used later in training.py)
    train_mask = df["year"] <= 2022
    thresholds = compute_thresholds(df[train_mask])

    os.makedirs("data/processed", exist_ok=True)
    thresholds.to_csv(THRESH_PATH, index=False)

    df = apply_labels(df, thresholds)

    print("Bust rate overall:", df["bust"].mean().round(4))
    print(df.groupby("lead_day")["bust"].mean().round(3))

    keep_cols = [
        "init_date", "valid_date", "lead_day", "lat", "lon", "region", "season",
        "month", "day_of_year", "year", "event_type",
        "forecast_t2m", "forecast_msl", "forecast_rain", "forecast_u10", "forecast_v10",
        "forecast_rh", "forecast_z500", "forecast_u850", "forecast_v850", "forecast_q850",
        "forecast_wind_speed_10m", "forecast_wind_speed_850",
        "nbhd_rain_mean", "nbhd_rain_std", "nbhd_msl_std",
        "abs_error_rain", "abs_error_t2m", "abs_error_msl",  # kept for evaluation/analysis only
        "bust",
    ]
    df[keep_cols].to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
