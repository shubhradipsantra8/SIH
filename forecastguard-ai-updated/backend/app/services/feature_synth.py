"""
feature_synth.py

The trained model expects a forecast FEATURE VECTOR (forecast_rain,
forecast_t2m, forecast_msl, ...) for a given lat/lon/date/lead_day. In a
production deployment those numbers come straight off the latest NCMRWF NWP
run. In this prototype there is no live NWP feed wired up, so this module
synthesizes a physically-structured stand-in forecast using the SAME
climatology logic as ml/generate_sample_data.py (seasonal rainfall, monsoon
depression / cyclone / western-disturbance episodes, lead-time error growth)
so the demo is internally consistent.

Everything downstream of this (the trained model, calibration, SHAP/fallback
explanation, historical-analogue search) is real - only the raw "current
forecast" numbers fed into it are simulated here. Swap `synthesize_forecast`
for a real NWP-feed reader and nothing else in the app needs to change.
"""

import hashlib
import numpy as np
from datetime import date as date_cls

ZONE_TO_REGION = {
    "North": "north", "West": "west", "Central": "central",
    "East": "east", "Northeast": "northeast", "South": "south",
}

STATE_META = {
    "Jammu and Kashmir": (33.5, 76.5, "North"),
    "Punjab": (31.0, 75.5, "North"),
    "Delhi": (28.6, 77.2, "North"),
    "Uttar Pradesh": (26.8, 80.9, "North"),
    "Rajasthan": (27.0, 74.2, "West"),
    "Gujarat": (22.3, 71.5, "West"),
    "Maharashtra": (19.7, 75.7, "West"),
    "Madhya Pradesh": (23.5, 78.5, "Central"),
    "Bihar": (25.5, 85.5, "East"),
    "West Bengal": (22.9, 87.5, "East"),
    "Odisha": (20.9, 85.1, "East"),
    "Assam": (26.2, 92.9, "Northeast"),
    "Karnataka": (15.3, 75.7, "South"),
    "Andhra Pradesh": (15.9, 79.7, "South"),
    "Tamil Nadu": (11.1, 78.6, "South"),
    "Kerala": (10.5, 76.3, "South"),
}


def _seeded_rng(*parts):
    key = "|".join(str(p) for p in parts)
    h = hashlib.sha256(key.encode()).hexdigest()
    seed = int(h[:8], 16)
    return np.random.default_rng(seed)


def season_of(d: date_cls) -> str:
    m = d.month
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "pre_monsoon"
    if m in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"


def draw_event_type(rng, season):
    p = rng.random()
    if season == "monsoon":
        if p < 0.10:
            return "monsoon_depression"
        if p < 0.16:
            return "cyclone"
    else:
        if p < 0.05:
            return "cyclone"
        if p < 0.12 and season in ("winter", "pre_monsoon"):
            return "western_disturbance"
    return "normal"


def synthesize_forecast(state: str, date_str: str, lead_day: int):
    if state not in STATE_META:
        raise ValueError(f"Unknown state: {state}")
    lat, lon, zone = STATE_META[state]
    region = ZONE_TO_REGION[zone]

    d = date_cls.fromisoformat(date_str)
    season = season_of(d)

    rng = _seeded_rng(state, date_str, lead_day)
    event_type = draw_event_type(rng, season)
    inst = rng.normal(0, 1) * (1.4 if event_type != "normal" else 0.6)
    lead_factor = np.sqrt(lead_day)

    t2m = 27 - 0.35 * (lat - 20) + rng.normal(0, 0.3)
    if season == "monsoon":
        t2m -= 2.0
    if season == "winter":
        t2m -= 4.0 if lat > 22 else 1.0

    msl = 1008 + rng.normal(0, 1.5) + rng.normal(inst * 0.5, 0.6 + 0.35 * lead_factor)

    rain_base = 2.0
    if season == "monsoon":
        rain_base = 18.0 if lon < 90 else 10.0
    if event_type in ("cyclone", "monsoon_depression"):
        rain_base += rng.uniform(40, 160)
    if event_type == "western_disturbance" and season in ("winter", "pre_monsoon"):
        rain_base += rng.uniform(10, 40)
    rain_true = max(0.0, rng.gamma(shape=1.3, scale=max(rain_base, 0.5)))
    forecast_rain = max(0.0, rain_true + rng.normal(inst * 6, (3 + 4 * lead_factor) * (1 + 0.9 * abs(inst))))

    u10 = rng.normal(2, 3) + rng.normal(0, (0.5 + 0.4 * lead_factor) * (1 + 0.6 * abs(inst)))
    v10 = rng.normal(1, 3) + rng.normal(0, (0.5 + 0.4 * lead_factor) * (1 + 0.6 * abs(inst)))
    if event_type in ("cyclone", "monsoon_depression"):
        u10 += rng.normal(0, 8)
        v10 += rng.normal(0, 8)

    rh = float(np.clip(55 + (25 if season == "monsoon" else 0) + rng.normal(0, 5 + 2 * lead_factor), 0, 100))
    z500 = 5820 + (lat - 20) * -6 + rng.normal(0, 4 + 2 * lead_factor)
    u850 = rng.normal(4, 4) + rng.normal(0, (0.5 + 0.4 * lead_factor))
    v850 = rng.normal(2, 4) + rng.normal(0, (0.5 + 0.4 * lead_factor))
    q850 = 6 + (6 if season == "monsoon" else 0) + rng.normal(0, 0.8 + 0.4 * lead_factor)

    nbhd_rain_mean = forecast_rain * rng.uniform(0.85, 1.15)
    nbhd_rain_std = abs(rng.normal(3 + 2 * lead_factor, 2)) * (1 + 0.5 * abs(inst))
    nbhd_msl_std = abs(rng.normal(1 + lead_factor * 0.5, 0.8)) * (1 + 0.5 * abs(inst))

    features = {
        "lead_day": lead_day, "lat": lat, "lon": lon,
        "month": d.month, "day_of_year": d.timetuple().tm_yday,
        "forecast_t2m": t2m, "forecast_msl": msl, "forecast_rain": forecast_rain,
        "forecast_u10": u10, "forecast_v10": v10, "forecast_rh": rh,
        "forecast_z500": z500, "forecast_u850": u850, "forecast_v850": v850,
        "forecast_q850": q850,
        "forecast_wind_speed_10m": float(np.sqrt(u10**2 + v10**2)),
        "forecast_wind_speed_850": float(np.sqrt(u850**2 + v850**2)),
        "nbhd_rain_mean": nbhd_rain_mean, "nbhd_rain_std": nbhd_rain_std,
        "nbhd_msl_std": nbhd_msl_std,
        "region": region, "season": season,
    }
    return features, event_type
