"""
generate_sample_data.py

IMPORTANT: This script generates a STAND-IN dataset that mimics the structure
and statistical behaviour of (NWP forecast) x (ERA5 verification) pairs over
an India grid, for multiple years, lead days (1-10) and seasons.

It exists so the REST of the pipeline (alignment -> error -> bust labels ->
features -> training -> calibration -> SHAP -> API) is fully runnable and
testable without network access or CDS/NCMRWF credentials.

When real credentials are available, replace this step with:
  - ml/download_era5_real.py   (Copernicus CDS API - verification/reanalysis)
  - your NCMRWF / WeatherBench2 forecast download step
and keep the SAME output schema so nothing downstream needs to change.

Output schema (data/raw/synthetic_forecast_obs.csv), one row per
(init_date, lead_day, lat, lon):

    init_date, valid_date, lead_day, lat, lon,
    forecast_t2m, forecast_msl, forecast_rain, forecast_u10, forecast_v10,
    forecast_rh, forecast_z500, forecast_u850, forecast_v850, forecast_q850,
    actual_t2m, actual_msl, actual_rain, actual_u10, actual_v10,
    actual_rh, actual_z500, actual_u850, actual_v850, actual_q850,
    event_type

Design choices that make this "physically plausible" rather than pure noise:
  - error variance grows with lead_day (medium range forecasts degrade)
  - monsoon season (Jun-Sep) has heavier rainfall + larger rainfall error
  - a small fraction of days are flagged as "active weather events"
    (synthetic cyclone / monsoon depression / western disturbance) which
    have systematically larger errors -> these become most of the true busts
  - spatial smoothness: nearby grid cells share correlated error via a
    shared per-region latent "instability" field
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import os

RNG = np.random.default_rng(42)

# ---- India grid (kept modest resolution for a runnable prototype) ----
LATS = np.arange(8, 34, 3.0)     # ~9 rows
LONS = np.arange(70, 97, 3.0)    # ~9 cols
GRID = [(round(la, 2), round(lo, 2)) for la in LATS for lo in LONS]

LEAD_DAYS = list(range(1, 11))
START_DATE = date(2019, 1, 1)
END_DATE = date(2024, 12, 31)
INIT_STRIDE_DAYS = 5   # one forecast run every 5 days keeps the file small


def season_of(d: date) -> str:
    m = d.month
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "pre_monsoon"
    if m in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"


def daterange(d0, d1, step):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=step)


def region_of(lat, lon):
    if lat > 26 and lon < 80:
        return "north"
    if lat > 24 and lon >= 88:
        return "northeast"
    if lat <= 15:
        return "south"
    if lon < 76:
        return "west"
    if lon >= 84:
        return "east"
    return "central"


def base_climatology(lat, lon, season, event_type):
    """Deterministic 'true atmospheric state' the forecast is trying to predict."""
    t2m = 27 - 0.35 * (lat - 20) + RNG.normal(0, 0.3)
    if season == "monsoon":
        t2m -= 2.0
    if season == "winter":
        t2m -= 4.0 if lat > 22 else 1.0

    msl = 1008 + RNG.normal(0, 1.5)
    rh = 55 + (25 if season == "monsoon" else 0) + RNG.normal(0, 5)

    rain_base = 2.0
    if season == "monsoon":
        rain_base = 18.0 if lon < 90 else 10.0
    if event_type in ("cyclone", "monsoon_depression"):
        rain_base += RNG.uniform(40, 160)
    if event_type == "western_disturbance" and season in ("winter", "pre_monsoon"):
        rain_base += RNG.uniform(10, 40)
    rain = max(0.0, RNG.gamma(shape=1.3, scale=max(rain_base, 0.5)))

    u10 = RNG.normal(2, 3)
    v10 = RNG.normal(1, 3)
    if event_type in ("cyclone", "monsoon_depression"):
        u10 += RNG.normal(0, 8)
        v10 += RNG.normal(0, 8)
        msl -= RNG.uniform(8, 25)

    z500 = 5820 + (lat - 20) * -6 + RNG.normal(0, 8)
    u850 = RNG.normal(4, 4)
    v850 = RNG.normal(2, 4)
    q850 = 6 + (6 if season == "monsoon" else 0) + RNG.normal(0, 1.2)

    return dict(t2m=t2m, msl=msl, rain=rain, u10=u10, v10=v10, rh=rh,
                z500=z500, u850=u850, v850=v850, q850=q850)


def draw_event_type(season):
    p = RNG.random()
    if season == "monsoon":
        if p < 0.06:
            return "monsoon_depression"
        if p < 0.09:
            return "cyclone"
    else:
        if p < 0.03:
            return "cyclone"
        if p < 0.08 and season in ("winter", "pre_monsoon"):
            return "western_disturbance"
    return "normal"


def main():
    rows = []
    init_dates = list(daterange(START_DATE, END_DATE, INIT_STRIDE_DAYS))

    for init_date in init_dates:
        season = season_of(init_date)
        event_type = draw_event_type(season)
        # shared latent "instability field" per init_date -> spatial correlation
        instability = {}
        for (lat, lon) in GRID:
            reg = region_of(lat, lon)
            instability[(lat, lon)] = RNG.normal(0, 1) * (
                1.4 if event_type != "normal" else 0.6
            )

        for lead_day in LEAD_DAYS:
            valid_date = init_date + timedelta(days=lead_day)
            # error growth with lead time (sqrt growth is a common heuristic)
            lead_factor = np.sqrt(lead_day)

            for (lat, lon) in GRID:
                truth = base_climatology(lat, lon, season, event_type)
                inst = instability[(lat, lon)]

                # forecast = truth + structured error that grows with lead
                # time and is amplified by local instability / active events
                err_scale_rain = (3 + 4 * lead_factor) * (1 + 0.9 * abs(inst))
                err_scale_temp = (0.4 + 0.25 * lead_factor) * (1 + 0.3 * abs(inst))
                err_scale_msl = (0.6 + 0.35 * lead_factor) * (1 + 0.5 * abs(inst))
                err_scale_wind = (0.5 + 0.4 * lead_factor) * (1 + 0.6 * abs(inst))

                f_rain = max(0.0, truth["rain"] + RNG.normal(inst * 6, err_scale_rain))
                f_t2m = truth["t2m"] + RNG.normal(-inst * 0.3, err_scale_temp)
                f_msl = truth["msl"] + RNG.normal(inst * 0.5, err_scale_msl)
                f_u10 = truth["u10"] + RNG.normal(0, err_scale_wind)
                f_v10 = truth["v10"] + RNG.normal(0, err_scale_wind)
                f_rh = np.clip(truth["rh"] + RNG.normal(0, 5 + 2 * lead_factor), 0, 100)
                f_z500 = truth["z500"] + RNG.normal(0, 4 + 2 * lead_factor)
                f_u850 = truth["u850"] + RNG.normal(0, err_scale_wind)
                f_v850 = truth["v850"] + RNG.normal(0, err_scale_wind)
                f_q850 = truth["q850"] + RNG.normal(0, 0.8 + 0.4 * lead_factor)

                rows.append(dict(
                    init_date=init_date.isoformat(),
                    valid_date=valid_date.isoformat(),
                    lead_day=lead_day,
                    lat=lat, lon=lon,
                    region=region_of(lat, lon),
                    season=season,
                    event_type=event_type,
                    forecast_t2m=f_t2m, forecast_msl=f_msl, forecast_rain=f_rain,
                    forecast_u10=f_u10, forecast_v10=f_v10, forecast_rh=f_rh,
                    forecast_z500=f_z500, forecast_u850=f_u850,
                    forecast_v850=f_v850, forecast_q850=f_q850,
                    actual_t2m=truth["t2m"], actual_msl=truth["msl"],
                    actual_rain=truth["rain"], actual_u10=truth["u10"],
                    actual_v10=truth["v10"], actual_rh=truth["rh"],
                    actual_z500=truth["z500"], actual_u850=truth["u850"],
                    actual_v850=truth["v850"], actual_q850=truth["q850"],
                ))

    df = pd.DataFrame(rows)
    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/synthetic_forecast_obs.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(df["event_type"].value_counts())


if __name__ == "__main__":
    main()
