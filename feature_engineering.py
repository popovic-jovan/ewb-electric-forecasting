# build_electricity_datasets.py
# ------------------------------------------------------------
# Create three model-ready datasets from merged_electricity_weather.csv
#   - Time-series   -> dataset_timeseries.csv
#   - Machine Learn -> dataset_ml.csv
#   - Deep Learn    -> dataset_dl.csv
#
# Adds:
#   - WA holidays (2022–2023)
#   - DOW + cyclical encodings (hour/month/dow)
#   - Daypart flags (night, 5–9pm)
#   - Weather-derived features (temp_mean, temp_range, is_hot_day, is_cold_night)
#   - Short-term memory (lag1, lag24, roll3, roll24)
#   - NEW: Cooling (CDH) @ base temp (default 18°C)
# ------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, List

# --------------------
# CONFIG (edit if needed)
# --------------------
INPUT_CSV   = "/Users/jovanpopovic/Desktop/honours_project/_merged_output/merged_electricity_weather.csv"
OUTPUT_DIR  = "."

# Degree-hour base temperature (°C)
BASE_TEMP_DEG_C = 24.0

# Peak/shoulder definitions (edit to match tariff if desired)
NIGHT_HOURS = set(list(range(0, 7)) + [22, 23])  # 00–06 & 22–23
PEAK_HOURS  = {17, 18, 19, 20}                    # 5–9 pm

# Temperature thresholds (°C)
HOT_DAY_THRESHOLD    = 35.0   # max temp ≥ 35°C
COLD_NIGHT_THRESHOLD = 15.0    # min temp ≤ 15°C

# WA Public Holidays (2022–2023), incl. observed days
WA_HOLIDAYS_2022 = [
    "2022-04-25","2022-06-06","2022-09-22","2022-09-26","2022-12-25","2022-12-26","2022-12-27"
]
WA_HOLIDAYS_2023 = [
    "2023-01-01","2023-01-02","2023-01-26","2023-03-06","2023-04-07","2023-04-10",
    "2023-04-25"
]
WA_HOLIDAYS = set(pd.to_datetime(WA_HOLIDAYS_2022 + WA_HOLIDAYS_2023))

# Columns we plan to drop (redundant/low-value/constant)
DROP_COLS = [
    "ref","row","aggregate_date","date","time","quarter",
    "error_check_day","error_check_hour","received_value",
    "period_over_which_rainfall_was_measured_days",
    "days_of_accumulation_of_minimum_temperature",
    "days_of_accumulation_of_maximum_temperature",
]

# Weather column names
COL_MAX_T = "maximum_temperature_degree_c"
COL_MIN_T = "minimum_temperature_degree_c"
COL_RAIN  = "rainfall_amount_millimetres"
COL_SOLAR = "daily_global_solar_exposure_mj_m_m"

TARGET_COL = "delivered_value"

# --------------------
# Helpers
# --------------------
def build_timestamp(df: pd.DataFrame) -> pd.Series:
    if not all(c in df.columns for c in ["aggregate_year","aggregate_month","aggregate_day"]):
        raise ValueError("Expected columns: aggregate_year, aggregate_month, aggregate_day")
    hour_col = "aggregate_hour" if "aggregate_hour" in df.columns else ("hour" if "hour" in df.columns else None)
    if hour_col is None:
        raise ValueError("Expected either 'aggregate_hour' or 'hour'")
    return pd.to_datetime(dict(
        year=df["aggregate_year"].astype(int),
        month=df["aggregate_month"].astype(int),
        day=df["aggregate_day"].astype(int),
        hour=df[hour_col].astype(int)
    ))

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date_only"]     = df["timestamp"].dt.normalize()
    df["is_holiday_wa"] = df["date_only"].isin(WA_HOLIDAYS).astype(int)
    df["is_weekend"]    = df["timestamp"].dt.dayofweek.isin([5,6]).astype(int)

    df["hour"]  = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["dow"]   = df["timestamp"].dt.dayofweek  # 0=Mon ... 6=Sun

    # Cyclical encodings
    df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["dow_sin"]   = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]   = np.cos(2*np.pi*df["dow"]/7)

    # Daypart flags
    df["is_night"]       = df["hour"].isin(NIGHT_HOURS).astype(int)
    df["is_peak_5to9pm"] = df["hour"].isin(PEAK_HOURS).astype(int)
    return df

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    have_max = COL_MAX_T in df.columns
    have_min = COL_MIN_T in df.columns

    if have_max and have_min:
        df["temp_mean"]   = (df[COL_MAX_T] + df[COL_MIN_T]) / 2.0
        df["temp_range"]  = (df[COL_MAX_T] - df[COL_MIN_T])
        df["is_hot_day"]   = (df[COL_MAX_T] >= HOT_DAY_THRESHOLD).astype(int)
        df["is_cold_night"] = (df[COL_MIN_T] <= COLD_NIGHT_THRESHOLD).astype(int)
        # NEW: Degree Hours (hourly)
        base = float(BASE_TEMP_DEG_C)
        df["CDH"] = np.maximum(0.0, df["temp_mean"] - base)   # Cooling Degree Hours
    else:
        df["temp_mean"]   = np.nan
        df["temp_range"]  = np.nan
        df["is_hot_day"]  = 0
        df["is_cold_night"]= 0
        df["CDH"] = 0.0


    return df

def add_short_memory_features(df: pd.DataFrame, group_key: Optional[str]) -> pd.DataFrame:
    def _apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp")
        g["delivered_lag1"]  = g[TARGET_COL].shift(1)
        g["delivered_lag24"] = g[TARGET_COL].shift(24)
        g["roll3_mean"]   = g[TARGET_COL].rolling(window=3,  min_periods=1).mean()
        g["roll24_mean"]  = g[TARGET_COL].rolling(window=24, min_periods=1).mean()
        g["roll24_sum"]   = g[TARGET_COL].rolling(window=24, min_periods=1).sum()
        return g
    return df.groupby(group_key, group_keys=False).apply(_apply) if group_key else _apply(df)

def ensure_columns_exist(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

# --------------------
# Main
# --------------------
def main():
    in_path = Path(INPUT_CSV)
    if not in_path.exists():
        raise FileNotFoundError(f"Could not find {in_path.resolve()}")

    df = pd.read_csv(in_path)

    # Build timestamp
    df["timestamp"] = build_timestamp(df)

    # Drop redundant/low-value columns
    df = df.drop(columns=ensure_columns_exist(df, DROP_COLS), errors="ignore")

    # Calendar + cyclical
    df = add_calendar_features(df)

    # Weather-derived + degree hours
    df = add_weather_features(df)

    # Sort + group key for per-meter lags
    group_key = "meter_ui" if "meter_ui" in df.columns else ("nmi_ui" if "nmi_ui" in df.columns else None)
    sort_cols = [group_key, "timestamp"] if group_key else ["timestamp"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Lags/rolling
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataset.")
    df = add_short_memory_features(df, group_key=group_key)

    # ----- Column sets -----
    id_cols      = ensure_columns_exist(df, ["meter_ui","nmi_ui"])
    weather_cols = ensure_columns_exist(df, [COL_MAX_T, COL_MIN_T, COL_RAIN, COL_SOLAR])

    common_cols = [
        "timestamp", TARGET_COL,
        "hour","month","dow",
        "hour_sin","hour_cos","month_sin","month_cos","dow_sin","dow_cos",
        "is_weekend","is_holiday_wa","is_night","is_peak_5to9pm",
        "temp_mean","temp_range","is_hot_day","is_cold_night",
        # NEW degree-hour features
        "CDH",
    ]

    # Time-series (no explicit lags/rolls)
    ts_cols = id_cols + common_cols + weather_cols
    ts_df = df[ts_cols].copy()

    # Machine Learning (lags/rolls included; drop initial NaNs)
    ml_cols = id_cols + common_cols + weather_cols + ["delivered_lag1","delivered_lag24","roll3_mean","roll24_mean","roll24_sum"]
    ml_df = df[ml_cols].copy()
    ml_df = ml_df.dropna(subset=["delivered_lag1","delivered_lag24"]).reset_index(drop=True)

    # Deep Learning (lags included; NaNs dropped for convenience)
    dl_cols = id_cols + common_cols + weather_cols + ["delivered_lag1","delivered_lag24"]
    dl_df = df[dl_cols].copy()
    dl_df = dl_df.dropna(subset=["delivered_lag1","delivered_lag24"]).reset_index(drop=True)

    # Optional: integer meter indices for embeddings
    if "meter_ui" in dl_df.columns:
        meter_map = {m:i for i, m in enumerate(sorted(df["meter_ui"].dropna().unique()))}
        dl_df["meter_idx"] = dl_df["meter_ui"].map(meter_map)
    elif "nmi_ui" in dl_df.columns:
        meter_map = {m:i for i, m in enumerate(sorted(df["nmi_ui"].dropna().unique()))}
        dl_df["meter_idx"] = dl_df["nmi_ui"].map(meter_map)

    # Save
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_path = out_dir / "dataset_timeseries.csv"
    ml_path = out_dir / "dataset_ml.csv"
    dl_path = out_dir / "dataset_dl.csv"
    ts_df.to_csv(ts_path, index=False)
    ml_df.to_csv(ml_path, index=False)
    dl_df.to_csv(dl_path, index=False)

    print(f"Base temp (°C) for CDH= {BASE_TEMP_DEG_C}")
    print(f"Saved:\n  - {ts_path}\n  - {ml_path}\n  - {dl_path}")

if __name__ == "__main__":
    main()
