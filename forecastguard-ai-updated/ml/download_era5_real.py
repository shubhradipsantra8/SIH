"""
download_era5_real.py

REAL data download script (requires a free Copernicus CDS account + API key
in ~/.cdsapirc). This is what you swap in to replace
generate_sample_data.py's synthetic ERA5 half once you're ready to move off
the prototype dataset.

This alone is NOT sufficient: ERA5 here is your VERIFICATION/ground-truth
source. You still need a historical NWP FORECAST source (e.g. an NCMRWF
archive, or a WeatherBench2-compatible forecast dataset) to compute
forecast-minus-actual error. See README.md section "Data strategy".

Usage:
    pip install cdsapi
    # create ~/.cdsapirc with your UID and API key from
    # https://cds.climate.copernicus.eu/api-how-to
    python ml/download_era5_real.py --year 2023 --month 07
"""

import argparse
import os

INDIA_AREA = [38, 68, 6, 98]  # North, West, South, East

SURFACE_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

PRESSURE_LEVEL_VARIABLES = [
    "temperature",
    "specific_humidity",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
]
PRESSURE_LEVELS = ["250", "500", "850"]


def download_surface(year: str, month: str, out_dir: str):
    import cdsapi
    client = cdsapi.Client()
    request = {
        "product_type": ["reanalysis"],
        "variable": SURFACE_VARIABLES,
        "year": [year],
        "month": [month],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "area": INDIA_AREA,
    }
    out_path = os.path.join(out_dir, f"era5_surface_{year}_{month}.nc")
    client.retrieve("reanalysis-era5-single-levels", request, out_path)
    print(f"Wrote {out_path}")


def download_pressure_levels(year: str, month: str, out_dir: str):
    import cdsapi
    client = cdsapi.Client()
    request = {
        "product_type": ["reanalysis"],
        "variable": PRESSURE_LEVEL_VARIABLES,
        "pressure_level": PRESSURE_LEVELS,
        "year": [year],
        "month": [month],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00", "12:00"],
        "data_format": "netcdf",
        "area": INDIA_AREA,
    }
    out_path = os.path.join(out_dir, f"era5_pressure_{year}_{month}.nc")
    client.retrieve("reanalysis-era5-pressure-levels", request, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--out_dir", default="data/raw")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    download_surface(args.year, args.month, args.out_dir)
    download_pressure_levels(args.year, args.month, args.out_dir)
