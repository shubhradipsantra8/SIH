"""
download_real_forecast_era5.py

Fast/vectorized WeatherBench2 HRES + ERA5 downloader
for ForecastGuard AI.

Forecast:
    ECMWF IFS HRES via WeatherBench2

Verification:
    ERA5 via WeatherBench2

IMPORTANT:
    HRES is ECMWF, NOT IMD/NCMRWF.
    Use this as the real research/prototype NWP source.
"""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# DATA SOURCES
# ============================================================

FORECAST_URL = (
    "gs://weatherbench2/datasets/hres/"
    "2016-2022-0012-1440x721.zarr"
)

ERA5_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)


# ============================================================
# INDIA TEST GRID
# ============================================================
# Keep this sparse for the first successful test.
# Later we can increase resolution to 0.5° / 0.25°.

LATS = np.arange(8, 34, 3.0)
LONS = np.arange(70, 97, 3.0)


# ============================================================
# LEAD DAYS
# ============================================================

LEAD_DAYS = list(range(1, 11))

PRESSURE_LEVEL_850 = 850
PRESSURE_LEVEL_500 = 500


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def season_of(ts):
    """Return meteorological season category."""

    month = pd.Timestamp(ts).month

    if month in (12, 1, 2):
        return "winter"

    if month in (3, 4, 5):
        return "pre_monsoon"

    if month in (6, 7, 8, 9):
        return "monsoon"

    return "post_monsoon"


def region_of(lat, lon):
    """Simple India regional classification."""

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


def specific_humidity_to_rh(q, temperature_kelvin, pressure_hpa):
    """
    Approximate relative humidity (%) from specific humidity.

    This is a DERIVED quantity.
    It is not a native HRES RH variable.
    """

    temperature_celsius = temperature_kelvin - 273.15

    # Saturation vapor pressure
    es = (
        6.112
        * np.exp(
            (17.67 * temperature_celsius)
            / (temperature_celsius + 243.5)
        )
    )

    # Actual vapor pressure
    e = (
        q * pressure_hpa
        / (0.622 + 0.378 * q)
    )

    rh = 100.0 * e / es

    return np.clip(rh, 0, 100)


def event_type_from_error(rain_error_mm, msl_hpa):
    """
    Metadata only.

    IMPORTANT:
    This is NOT the ML bust label.
    Real bust labels will be generated later
    by build_features_and_labels.py.
    """

    if msl_hpa < 990:
        return "cyclone"

    if rain_error_mm > 60:
        return "monsoon_depression"

    return "normal"


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        required=True,
        help="Start initialization date YYYY-MM-DD",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="End initialization date YYYY-MM-DD",
    )

    parser.add_argument(
        "--init_stride_days",
        type=int,
        default=10,
        help="Use every N days of forecast initializations",
    )

    parser.add_argument(
        "--out",
        default="data/raw/real_forecast_obs.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    # ========================================================
    # OPEN DATASETS
    # ========================================================

    print()
    print("=" * 70)
    print("ForecastGuard AI - Real WeatherBench2 Downloader")
    print("=" * 70)
    print()

    print("Opening WeatherBench2 forecast store...")

    fc = xr.open_zarr(
        FORECAST_URL,
        chunks="auto",
        storage_options={"token": "anon"},
    )

    print("Opening ERA5 verification store...")

    era5 = xr.open_zarr(
        ERA5_URL,
        chunks="auto",
        storage_options={"token": "anon"},
    )

    print()
    print("Datasets opened successfully.")
    print()

    # ========================================================
    # SHOW DATASET INFORMATION
    # ========================================================

    print("Forecast dimensions:")
    print(fc.dims)

    print()
    print("Forecast variables:")
    print(list(fc.data_vars))

    print()
    print("ERA5 variables:")
    print(list(era5.data_vars))

    print()

    # ========================================================
    # SELECT INDIA GRID
    # ========================================================

    print("Selecting India grid...")

    fc_india = fc.sel(
        latitude=LATS,
        longitude=LONS,
        method="nearest",
    )

    fc_india = fc_india.sel(
        time=slice(args.start, args.end)
    )

    era5_india = era5.sel(
        latitude=LATS,
        longitude=LONS,
        method="nearest",
    )

    # ========================================================
    # SELECT INITIALIZATION TIMES
    # ========================================================

    # HRES contains 00 and 12 UTC initializations.
    #
    # Therefore:
    #
    # 1 day = 2 initialization steps
    #
    # Example:
    # init_stride_days=10
    # → every 20 initialization steps

    stride_steps = max(
        1,
        args.init_stride_days * 2
    )

    init_times = fc_india.time.values[::stride_steps]

    print()
    print(f"Selected initialization times: {len(init_times)}")
    print(f"Grid latitude points: {len(LATS)}")
    print(f"Grid longitude points: {len(LONS)}")
    print(f"Grid points: {len(LATS) * len(LONS)}")
    print(f"Lead days: {LEAD_DAYS}")

    estimated_rows = (
        len(init_times)
        * len(LEAD_DAYS)
        * len(LATS)
        * len(LONS)
    )

    print(f"Estimated output rows: {estimated_rows:,}")
    print()

    # ========================================================
    # OUTPUT STORAGE
    # ========================================================

    rows = []

    # ========================================================
    # PROCESS EACH INITIALIZATION
    # ========================================================

    for init_index, init_time in enumerate(
        init_times,
        start=1,
    ):

        init_ts = pd.Timestamp(init_time)

        print()
        print(
            f"[{init_index}/{len(init_times)}] "
            f"Processing initialization: {init_ts}"
        )

        # ----------------------------------------------------
        # VALID TIMES
        # ----------------------------------------------------

        valid_times = [
            init_ts + pd.Timedelta(days=lead)
            for lead in LEAD_DAYS
        ]

        # ----------------------------------------------------
        # CREATE LEAD-TIME COORDINATE
        # ----------------------------------------------------

        lead_timedelta = xr.DataArray(
            pd.to_timedelta(
                LEAD_DAYS,
                unit="D",
            ).to_numpy(),
            dims="lead_day",
            coords={
                "lead_day": LEAD_DAYS
            },
        )

        # ----------------------------------------------------
        # BATCH FORECAST READ
        # ----------------------------------------------------

        try:

            print("  Loading forecast block...")

            forecast_block = fc_india.sel(
                time=init_ts,
                prediction_timedelta=lead_timedelta,
                method="nearest",
            )

            forecast_block = forecast_block.load()

            print("  Forecast block loaded.")

        except Exception as error:

            print()
            print("  ERROR loading forecast:")
            print(
                f"  {type(error).__name__}: {error}"
            )

            continue

        # ----------------------------------------------------
        # BATCH ERA5 READ
        # ----------------------------------------------------

        try:

            print("  Loading ERA5 verification block...")

            valid_time_array = xr.DataArray(
                np.array(
                    valid_times,
                    dtype="datetime64[ns]",
                ),
                dims="lead_day",
                coords={
                    "lead_day": LEAD_DAYS
                },
            )

            actual_block = era5_india.sel(
                time=valid_time_array,
                method="nearest",
            )

            actual_block = actual_block.load()

            print("  ERA5 block loaded.")

        except Exception as error:

            print()
            print("  ERROR loading ERA5:")

            print(
                f"  {type(error).__name__}: {error}"
            )

            continue

        # ====================================================
        # FORECAST VARIABLES
        # ====================================================

        try:

            print("  Extracting forecast variables...")

            f_t2m = (
                forecast_block[
                    "2m_temperature"
                ].values
                - 273.15
            )

            f_msl = (
                forecast_block[
                    "mean_sea_level_pressure"
                ].values
                / 100.0
            )

            f_rain = (
                forecast_block[
                    "total_precipitation_6hr"
                ].values
                * 1000.0
            )

            f_u10 = forecast_block[
                "10m_u_component_of_wind"
            ].values

            f_v10 = forecast_block[
                "10m_v_component_of_wind"
            ].values

            f_z500 = (
                forecast_block[
                    "geopotential"
                ]
                .sel(
                    level=PRESSURE_LEVEL_500
                )
                .values
                / 9.80665
            )

            f_u850 = (
                forecast_block[
                    "u_component_of_wind"
                ]
                .sel(
                    level=PRESSURE_LEVEL_850
                )
                .values
            )

            f_v850 = (
                forecast_block[
                    "v_component_of_wind"
                ]
                .sel(
                    level=PRESSURE_LEVEL_850
                )
                .values
            )

            f_t850 = (
                forecast_block[
                    "temperature"
                ]
                .sel(
                    level=PRESSURE_LEVEL_850
                )
                .values
            )

            f_q850 = (
                forecast_block[
                    "specific_humidity"
                ]
                .sel(
                    level=PRESSURE_LEVEL_850
                )
                .values
            )

            f_rh = specific_humidity_to_rh(
                f_q850,
                f_t850,
                PRESSURE_LEVEL_850,
            )

        except Exception as error:

            print()
            print("  ERROR extracting forecast variables:")
            print(
                f"  {type(error).__name__}: {error}"
            )

            continue

        # ====================================================
        # ACTUAL / ERA5 VARIABLES
        # ====================================================

        try:

            print("  Extracting ERA5 variables...")

            a_t2m = (
                actual_block[
                    "2m_temperature"
                ].values
                - 273.15
            )

            a_msl = (
                actual_block[
                    "mean_sea_level_pressure"
                ].values
                / 100.0
            )

            a_rain = (
                actual_block[
                    "total_precipitation_6hr"
                ].values
                * 1000.0
            )

            a_u10 = actual_block[
                "10m_u_component_of_wind"
            ].values

            a_v10 = actual_block[
                "10m_v_component_of_wind"
            ].values

            a_z500 = (
                actual_block[
                    "geopotential"
                ]
                .sel(
                    level=PRESSURE_LEVEL_500
                )
                .values
                / 9.80665
            )

            a_u850 = (
                actual_block[
                    "u_component_of_wind"
                ]
                .sel(
                    level=PRESSURE_LEVEL_850
                )
                .values
            )

            a_v850 = (
                actual_block[
                    "v_component_of_wind"
                ]
                .sel(
                    level=PRESSURE_LEVEL_850
                )
                .values
            )

            a_t850 = (
                actual_block[
                    "temperature"
                ]
                .sel(
                    level=PRESSURE_LEVEL_850
                )
                .values
            )

            a_q850 = (
                actual_block[
                    "specific_humidity"
                ]
                .sel(
                    level=PRESSURE_LEVEL_850
                )
                .values
            )

            a_rh = specific_humidity_to_rh(
                a_q850,
                a_t850,
                PRESSURE_LEVEL_850,
            )

        except Exception as error:

            print()
            print("  ERROR extracting ERA5 variables:")
            print(
                f"  {type(error).__name__}: {error}"
            )

            continue

        # ====================================================
        # CHECK SHAPES
        # ====================================================

        expected_shape = (
            len(LEAD_DAYS),
            len(LATS),
            len(LONS),
        )

        check_arrays = {
            "forecast_t2m": f_t2m,
            "forecast_msl": f_msl,
            "forecast_rain": f_rain,
            "forecast_u10": f_u10,
            "forecast_v10": f_v10,
            "forecast_rh": f_rh,
            "forecast_z500": f_z500,
            "forecast_u850": f_u850,
            "forecast_v850": f_v850,
            "forecast_q850": f_q850,

            "actual_t2m": a_t2m,
            "actual_msl": a_msl,
            "actual_rain": a_rain,
            "actual_u10": a_u10,
            "actual_v10": a_v10,
            "actual_rh": a_rh,
            "actual_z500": a_z500,
            "actual_u850": a_u850,
            "actual_v850": a_v850,
            "actual_q850": a_q850,
        }

        bad_shapes = []

        for name, array in check_arrays.items():

            if np.asarray(array).shape != expected_shape:

                bad_shapes.append(
                    (
                        name,
                        np.asarray(array).shape,
                    )
                )

        if bad_shapes:

            print()
            print("ERROR: unexpected data shape.")

            for name, shape in bad_shapes:

                print(
                    f"  {name}: "
                    f"{shape} "
                    f"(expected {expected_shape})"
                )

            continue

        # ====================================================
        # CONVERT BATCH TO ROWS
        # ====================================================

        print("  Converting block to rows...")

        for lead_index, lead_day in enumerate(
            LEAD_DAYS
        ):

            valid_ts = valid_times[lead_index]

            season = season_of(init_ts)

            for lat_index, lat in enumerate(LATS):

                for lon_index, lon in enumerate(LONS):

                    values = [

                        f_t2m[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        f_msl[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        f_rain[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        f_u10[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        f_v10[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        f_rh[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        f_z500[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        f_u850[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        f_v850[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        f_q850[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        a_t2m[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        a_msl[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        a_rain[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        a_u10[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        a_v10[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        a_rh[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        a_z500[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        a_u850[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        a_v850[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],

                        a_q850[
                            lead_index,
                            lat_index,
                            lon_index,
                        ],
                    ]

                    # Skip invalid rows.
                    if not np.all(
                        np.isfinite(values)
                    ):
                        continue

                    forecast_rain_value = float(
                        f_rain[
                            lead_index,
                            lat_index,
                            lon_index,
                        ]
                    )

                    actual_rain_value = float(
                        a_rain[
                            lead_index,
                            lat_index,
                            lon_index,
                        ]
                    )

                    actual_msl_value = float(
                        a_msl[
                            lead_index,
                            lat_index,
                            lon_index,
                        ]
                    )

                    rows.append(
                        {

                            "init_date":
                                init_ts.date().isoformat(),

                            "valid_date":
                                valid_ts.date().isoformat(),

                            "lead_day":
                                lead_day,

                            "lat":
                                round(
                                    float(lat),
                                    2,
                                ),

                            "lon":
                                round(
                                    float(lon),
                                    2,
                                ),

                            "region":
                                region_of(
                                    lat,
                                    lon,
                                ),

                            "season":
                                season,

                            "event_type":
                                event_type_from_error(
                                    abs(
                                        forecast_rain_value
                                        - actual_rain_value
                                    ),
                                    actual_msl_value,
                                ),

                            # ----------------------------
                            # FORECAST
                            # ----------------------------

                            "forecast_t2m":
                                float(
                                    f_t2m[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "forecast_msl":
                                float(
                                    f_msl[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "forecast_rain":
                                forecast_rain_value,

                            "forecast_u10":
                                float(
                                    f_u10[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "forecast_v10":
                                float(
                                    f_v10[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "forecast_rh":
                                float(
                                    f_rh[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "forecast_z500":
                                float(
                                    f_z500[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "forecast_u850":
                                float(
                                    f_u850[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "forecast_v850":
                                float(
                                    f_v850[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "forecast_q850":
                                float(
                                    f_q850[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            # ----------------------------
                            # ACTUAL / ERA5
                            # ----------------------------

                            "actual_t2m":
                                float(
                                    a_t2m[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "actual_msl":
                                actual_msl_value,

                            "actual_rain":
                                actual_rain_value,

                            "actual_u10":
                                float(
                                    a_u10[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "actual_v10":
                                float(
                                    a_v10[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "actual_rh":
                                float(
                                    a_rh[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "actual_z500":
                                float(
                                    a_z500[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "actual_u850":
                                float(
                                    a_u850[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "actual_v850":
                                float(
                                    a_v850[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),

                            "actual_q850":
                                float(
                                    a_q850[
                                        lead_index,
                                        lat_index,
                                        lon_index,
                                    ]
                                ),
                        }
                    )

        print(
            f"  Finished initialization. "
            f"Total rows: {len(rows):,}"
        )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    if not rows:

        raise RuntimeError(
            "\nNo rows were generated.\n"
            "Check the errors printed above."
        )

    df = pd.DataFrame(rows)

    # ========================================================
    # SAVE
    # ========================================================

    output_directory = os.path.dirname(args.out)

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    df.to_csv(
        args.out,
        index=False,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        f"Output file: {args.out}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Missing values: "
        f"{int(df.isna().sum().sum()):,}"
    )

    print()
    print("Lead-day distribution:")
    print(
        df[
            "lead_day"
        ].value_counts().sort_index()
    )

    print()
    print("Date range:")
    print(
        df["init_date"].min(),
        "→",
        df["init_date"].max(),
    )

    print()
    print("Region distribution:")
    print(
        df["region"].value_counts()
    )

    print()
    print("Sample:")
    print(
        df.head()
    )

    print()
    print(
        "Real forecast + ERA5 dataset successfully created."
    )


if __name__ == "__main__":
    main()