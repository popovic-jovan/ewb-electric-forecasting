"""Model-specific dataset preparation pipeline.

This module reads the merged electricity and weather dataset and produces
tailored, model-ready datasets for the SARIMA, SARIMAX, XGBoost (full and
minimal ablation), and LSTM pipelines. The workflow is:

1. Apply global preprocessing (drop meter M34, parse timestamps, enforce hourly
   index, interpolate gaps).
2. Build shared calendar and weather-derived features.
3. Execute model-specific feature engineering and save the corresponding
   artefacts under `model_datasets/`.

Running the script without arguments generates all datasets:

    python dataset_engineering/feature_engineering.py --models all
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


INPUT_CSV = Path(r"C:\Users\Owner\honours_project\datasets\merged_electricity_weather.csv")
MODEL_DATA_DIR = Path("datasets")

TARGET_COL = "delivered_value"
USAGE_COL = "usage_kwh"

# Degree-hour base temperatures (deg C)
COOLING_BASE = 24.0
HEATING_BASE = 18.0

# Time-of-day buckets (hour integers)
NIGHT_HOURS = set(list(range(0, 7)) + [22, 23])
PEAK_HOURS = {17, 18, 19, 20}

# Extreme temperature thresholds (deg C)
HOT_DAY_THRESHOLD = 35.0
COLD_NIGHT_THRESHOLD = 15.0

# Western Australia public holidays (2022-2023)
WA_HOLIDAYS = pd.to_datetime([
    "2022-04-25", "2022-06-06", "2022-09-22", "2022-09-26",
    "2022-12-25", "2022-12-26", "2022-12-27",
    "2023-01-01", "2023-01-02", "2023-01-26", "2023-03-06",
    "2023-04-07", "2023-04-10", "2023-04-25",
]).tz_localize(None)

# Columns that never survive into model datasets
DROP_COLS = [
    "ref", "row", "aggregate_date", "date", "time", "quarter",
    "error_check_day", "error_check_hour", "received_value",
    "period_over_which_rainfall_was_measured_days",
    "days_of_accumulation_of_minimum_temperature",
    "days_of_accumulation_of_maximum_temperature",
]

AGGREGATE_COMPONENTS = [
    "aggregate_year", "aggregate_month", "aggregate_day", "aggregate_hour",
]

# Raw weather columns
COL_MAX_T = "maximum_temperature_degree_c"
COL_MIN_T = "minimum_temperature_degree_c"
COL_RAIN = "rainfall_amount_millimetres"
COL_SOLAR = "daily_global_solar_exposure_mj_m_m"

# --------------------------------------------------------------------------- #


class ModelType(str, Enum):
    SARIMA = "sarima"
    SARIMAX = "sarimax"
    XGBOOST = "xgboost"
    XGBOOST_DISCONNECT = "xgboost_disconnect"
    XGBOOST_MINIMAL = "xgboost_minimal"
    LSTM = "lstm"

    @classmethod
    def from_arg(cls, value: str) -> "ModelType":
        try:
            return cls(value.lower())
        except ValueError as exc:
            valid = ", ".join(m.value for m in cls)
            raise argparse.ArgumentTypeError(f"Unknown model '{value}'. Choose from: {valid}") from exc


@dataclass
class DatasetArtefact:
    path: Path
    description: str
    extra: Optional[Dict[str, object]] = None


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


def build_timestamp(df: pd.DataFrame) -> pd.Series:
    """Construct a timezone-naive hourly timestamp from the aggregate components."""
    cols = {"aggregate_year", "aggregate_month", "aggregate_day"}
    if not cols.issubset(df.columns):
        missing = ", ".join(sorted(cols.difference(df.columns)))
        raise ValueError(f"Missing required columns for timestamp creation: {missing}")

    if "aggregate_hour" in df.columns:
        hour_col = "aggregate_hour"
    elif "hour" in df.columns:
        hour_col = "hour"
    else:
        raise ValueError("Expected an 'aggregate_hour' or 'hour' column.")

    ts = pd.to_datetime(
        dict(
            year=df["aggregate_year"].astype(int),
            month=df["aggregate_month"].astype(int),
            day=df["aggregate_day"].astype(int),
            hour=df[hour_col].astype(int),
        ),
        errors="raise",
    )
    if isinstance(ts, pd.Series):
        tz = getattr(ts.dt, "tz", None)
        return ts if tz is None else ts.dt.tz_localize(None)

    tz = getattr(ts, "tz", None)
    return ts if tz is None else ts.tz_localize(None)


def detect_meter_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("meter_ui", "nmi_ui", "meter_id"):
        if col in df.columns:
            return col
    return None


def determine_season(month: int) -> str:
    if month in (12, 1, 2):
        return "summer"
    if month in (3, 4, 5):
        return "autumn"
    if month in (6, 7, 8):
        return "winter"
    return "spring"


def ensure_hourly_frequency(df: pd.DataFrame, meter_col: Optional[str]) -> pd.DataFrame:
    """Resample each meter (or the whole frame) to an hourly frequency."""
    def _resample(group: pd.DataFrame) -> pd.DataFrame:
        idx = pd.date_range(group.index.min(), group.index.max(), freq="h")
        group = group.reindex(idx)
        return group

    if meter_col:
        pieces = []
        for meter, group in df.groupby(meter_col, group_keys=False):
            resampled = _resample(group.sort_index())
            resampled[meter_col] = meter
            pieces.append(resampled)
        combined = pd.concat(pieces, axis=0)
        combined.index.name = df.index.name
        return combined

    resampled = _resample(df.sort_index())
    resampled.index.name = df.index.name
    return resampled


def interpolate_numeric(df: pd.DataFrame, meter_col: Optional[str]) -> pd.DataFrame:
    """Interpolate numeric features in a time-aware manner."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not len(numeric_cols):
        return df

    def _interp(group: pd.DataFrame) -> pd.DataFrame:
        interp = group.copy()
        interp[numeric_cols] = (
            interp[numeric_cols]
            .interpolate(method="time", limit_direction="both")
            .ffill()
            .bfill()
        )
        return interp

    if meter_col and meter_col in df.columns:
        pieces = []
        for _, group in df.groupby(meter_col, sort=False):
            pieces.append(_interp(group))
        combined = pd.concat(pieces, axis=0)
        combined = combined.sort_index()
        return combined
    return _interp(df)


def forward_fill_strings(df: pd.DataFrame, meter_col: Optional[str]) -> pd.DataFrame:
    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if meter_col and meter_col in object_cols:
        object_cols = [col for col in object_cols if col != meter_col]
    if not len(object_cols):
        return df

    filled = df.copy()
    if meter_col and meter_col in df.columns:
        filled[object_cols] = (
            filled.groupby(meter_col)[object_cols]
            .transform(lambda g: g.ffill().bfill())
        )
    else:
        filled[object_cols] = filled[object_cols].ffill().bfill()
    return filled


def add_calendar_features(df: pd.DataFrame) -> None:
    idx = df.index
    df["date_only"] = idx.normalize()
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["day_of_year"] = idx.dayofyear

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday_wa"] = df["date_only"].isin(WA_HOLIDAYS).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    df["is_night"] = df["hour"].isin(NIGHT_HOURS).astype(int)
    df["is_peak_5to9pm"] = df["hour"].isin(PEAK_HOURS).astype(int)

    season_map = {m: determine_season(m) for m in range(1, 13)}
    df["season"] = df["month"].map(season_map)


def add_weather_features(df: pd.DataFrame) -> None:
    have_max = COL_MAX_T in df.columns
    have_min = COL_MIN_T in df.columns

    if have_max and have_min:
        df["temp_mean"] = (df[COL_MAX_T] + df[COL_MIN_T]) / 2.0
        df["temp_range"] = df[COL_MAX_T] - df[COL_MIN_T]
        df["is_hot_day"] = (df[COL_MAX_T] >= HOT_DAY_THRESHOLD).astype(int)
        df["is_cold_night"] = (df[COL_MIN_T] <= COLD_NIGHT_THRESHOLD).astype(int)
    else:
        df["temp_mean"] = np.nan
        df["temp_range"] = np.nan
        df["is_hot_day"] = 0
        df["is_cold_night"] = 0

    df["CDH"] = np.maximum(0.0, df["temp_mean"] - COOLING_BASE)
    df["HDH"] = np.maximum(0.0, HEATING_BASE - df["temp_mean"])

    df["rain_log1p"] = np.log1p(np.clip(df.get(COL_RAIN, 0.0), a_min=0.0, a_max=None))
    df["solar_log1p"] = np.log1p(np.clip(df.get(COL_SOLAR, 0.0), a_min=0.0, a_max=None))
    df["hour_temp_weight"] = df["hour_sin"] * df["temp_mean"]


def add_group_lags(
    df: pd.DataFrame,
    column: str,
    lags: Iterable[int],
    group_key: Optional[str],
    prefix: Optional[str] = None,
) -> List[str]:
    created: List[str] = []
    target = df[column]
    name_prefix = prefix or column
    for lag in lags:
        lag_col = f"{name_prefix}_lag_{lag}"
        if group_key:
            df[lag_col] = df.groupby(group_key)[column].shift(lag)
        else:
            df[lag_col] = target.shift(lag)
        created.append(lag_col)
    return created


def add_group_rolling(
    df: pd.DataFrame,
    column: str,
    windows: Dict[str, int],
    group_key: Optional[str],
) -> List[str]:
    created: List[str] = []

    def _rolling(series: pd.Series, win: int, func: str) -> pd.Series:
        roller = series.rolling(window=win, min_periods=1)
        if func == "mean":
            return roller.mean()
        if func == "std":
            return roller.std()
        if func == "sum":
            return roller.sum()
        raise ValueError(f"Unsupported rolling function '{func}'")

    for label, (window, func) in windows.items():
        col_name = f"{column}_{label}"
        if group_key:
            df[col_name] = df.groupby(group_key)[column].transform(lambda s: _rolling(s, window, func))
        else:
            df[col_name] = _rolling(df[column], window, func)
        created.append(col_name)
    return created


def drop_multicollinear_features(
    features: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[str]]:
    corr = features.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drops = [col for col in upper.columns if any(upper[col] > threshold)]
    reduced = features.drop(columns=drops) if drops else features
    return reduced, drops


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Dataset builders
# --------------------------------------------------------------------------- #


def prepare_base_dataframe() -> Tuple[pd.DataFrame, Optional[str]]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if "timestamp" not in df.columns:
        df["timestamp"] = build_timestamp(df)

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in AGGREGATE_COMPONENTS if c in df.columns], errors="ignore")

    meter_col = detect_meter_column(df)
    if meter_col and "M34" in df[meter_col].values:
        df = df[df[meter_col] != "M34"]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")
    df.index.name = "timestamp"

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL])
    df[USAGE_COL] = df[TARGET_COL].astype(float)

    df = ensure_hourly_frequency(df, meter_col)
    df = forward_fill_strings(df, meter_col)
    df = interpolate_numeric(df, meter_col)

    add_calendar_features(df)
    add_weather_features(df)

    return df, meter_col


def prepare_sarima_dataset(
    df: pd.DataFrame,
    meter_col: Optional[str],
) -> DatasetArtefact:
    cols = [USAGE_COL]
    if meter_col:
        cols.append(meter_col)
    sarima_df = df[cols].copy()
    sarima_df["y"] = sarima_df[USAGE_COL]

    if meter_col:
        sarima_df["y_diff_1"] = sarima_df.groupby(meter_col)["y"].diff()
        sarima_df["y_diff_24"] = sarima_df.groupby(meter_col)["y"].diff(24)
    else:
        sarima_df["y_diff_1"] = sarima_df["y"].diff()
        sarima_df["y_diff_24"] = sarima_df["y"].diff(24)

    sarima_df = sarima_df.dropna(subset=["y"])
    sarima_df = sarima_df.reset_index()

    output_path = MODEL_DATA_DIR / "dataset_sarima.csv"
    sarima_df.to_csv(output_path, index=False)
    return DatasetArtefact(
        path=output_path,
        description="Hourly univariate series for SARIMA (with helper differences).",
        extra={"rows": int(len(sarima_df))},
    )


def smooth_exogenous(df: pd.DataFrame, columns: Sequence[str], meter_col: Optional[str]) -> Dict[str, str]:
    """Apply a 3-hour rolling mean to specified columns; return mapping of original->smoothed column names."""
    mapping: Dict[str, str] = {}
    for col in columns:
        smoothed = f"{col}_smooth"
        if meter_col:
            df[smoothed] = df.groupby(meter_col)[col].transform(
                lambda s: s.rolling(window=3, min_periods=1).mean()
            )
        else:
            df[smoothed] = df[col].rolling(window=3, min_periods=1).mean()
        mapping[col] = smoothed
    return mapping


def prepare_sarimax_dataset(
    df: pd.DataFrame,
    meter_col: Optional[str],
) -> DatasetArtefact:
    base_cols = [
        "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
        "is_weekend", "is_holiday_wa", "is_night", "is_peak_5to9pm",
        "hour_temp_weight",
        "temp_mean", "temp_range", "CDH", "HDH",
        "rain_log1p", "solar_log1p",
    ]
    available_cols = [c for c in base_cols if c in df.columns]

    features = df[available_cols].copy()

    smooth_map = smooth_exogenous(df, ["temp_mean", "temp_range", "rain_log1p", "solar_log1p"], meter_col)
    for original, smoothed in smooth_map.items():
        if smoothed in df.columns:
            features[smoothed] = df[smoothed]

    lag_targets = ["temp_mean", "temp_range", "rain_log1p", "solar_log1p", "CDH", "HDH"]
    lag_features: List[str] = []
    for col in lag_targets:
        if col in df.columns:
            lag_features.extend(add_group_lags(df, col, lags=[1, 24], group_key=meter_col))
    for lag_col in lag_features:
        features[lag_col] = df[lag_col]

    if meter_col:
        features[meter_col] = df[meter_col]
        features = features.groupby(meter_col, group_keys=False).apply(lambda g: g.ffill().bfill())
        features = features.drop(columns=[meter_col])
    else:
        features = features.ffill().bfill()
    features = features.fillna(0.0)

    reduced_features, dropped = drop_multicollinear_features(features)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(reduced_features.values)
    scaled_df = pd.DataFrame(
        scaled_values,
        index=reduced_features.index,
        columns=reduced_features.columns,
    )

    sarimax_df = df[[USAGE_COL]].copy()
    sarimax_df["y"] = sarimax_df[USAGE_COL]
    for col in scaled_df.columns:
        sarimax_df[col] = scaled_df[col]
    if meter_col:
        sarimax_df[meter_col] = df[meter_col]

    sarimax_df = sarimax_df.dropna(subset=["y"])
    sarimax_df = sarimax_df.reset_index()

    output_path = MODEL_DATA_DIR / "dataset_sarimax.csv"
    sarimax_df.to_csv(output_path, index=False)

    scaler_path = MODEL_DATA_DIR / "sarimax_scaler_stats.json"
    scaler_stats = {
        "feature_order": list(reduced_features.columns),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with scaler_path.open("w", encoding="utf-8") as f:
        json.dump(scaler_stats, f, indent=2)

    return DatasetArtefact(
        path=output_path,
        description="Hourly SARIMAX dataset with scaled exogenous regressors.",
        extra={
            "dropped_multicollinear": dropped,
            "scaler_stats": str(scaler_path),
            "features": reduced_features.columns.tolist(),
        },
    )


def prepare_xgboost_dataset(
    df: pd.DataFrame,
    meter_col: Optional[str],
) -> DatasetArtefact:
    working = df.copy()
    if meter_col and meter_col not in working.columns:
        meter_col = None

    lag_cols = add_group_lags(
        working,
        USAGE_COL,
        lags=[1, 2, 24, 48, 168],
        group_key=meter_col,
        prefix="usage",
    )

    rolling_specs = {
        "roll_mean_3h": (3, "mean"),
        "roll_std_24h": (24, "std"),
        "roll_sum_168h": (168, "sum"),
    }
    rolling_cols = add_group_rolling(
        working,
        USAGE_COL,
        {label: (win, func) for label, (win, func) in rolling_specs.items()},
        meter_col,
    )

    # Weather-derived features
    weather_lags: List[str] = []
    rain_features: List[str] = []
    tmax_features: List[str] = []

    if COL_MAX_T in df.columns:
        weather_lags.extend(add_group_lags(working, COL_MAX_T, [24], meter_col, prefix="tmax"))
        tmax_roll_cols = add_group_rolling(
            working,
            COL_MAX_T,
            {"tmax_mean_168h": (168, "mean")},
            meter_col,
        )
        tmax_prev_cols: List[str] = []
        for col in tmax_roll_cols:
            prev_col = f"{col}_prevday"
            if meter_col:
                working[prev_col] = working.groupby(meter_col)[col].shift(24)
            else:
                working[prev_col] = working[col].shift(24)
            tmax_prev_cols.append(prev_col)
        working = working.drop(columns=tmax_roll_cols, errors="ignore")
        tmax_features.extend(tmax_prev_cols)

    if COL_SOLAR in df.columns:
        weather_lags.extend(add_group_lags(working, COL_SOLAR, [24], meter_col, prefix="solar"))

    if COL_RAIN in df.columns:
        rain_features.extend(add_group_lags(working, COL_RAIN, [24, 72, 168], meter_col, prefix="rain"))
        rain_roll_cols = add_group_rolling(
            working,
            COL_RAIN,
            {"rain_cum_72h": (72, "sum"), "rain_cum_168h": (168, "sum")},
            meter_col,
        )
        rain_prev_cols: List[str] = []
        for col in rain_roll_cols:
            prev_col = f"{col}_prevday"
            if meter_col:
                working[prev_col] = working.groupby(meter_col)[col].shift(24)
            else:
                working[prev_col] = working[col].shift(24)
            rain_prev_cols.append(prev_col)
        working = working.drop(columns=rain_roll_cols, errors="ignore")
        rain_features.extend(rain_prev_cols)

    # Interactions
    if "is_weekend" in working.columns and "CDH" in working.columns:
        working["is_weekend_CDH"] = working["is_weekend"] * working["CDH"]

    categorical_cols: List[str] = []
    if meter_col:
        working["meter_code"] = working[meter_col].astype("category")
        categorical_cols.append("meter_code")

    # Prepare target shifted one hour ahead
    if meter_col:
        working["y"] = working.groupby(meter_col)[USAGE_COL].shift(-1)
    else:
        working["y"] = working[USAGE_COL].shift(-1)

    features_to_keep = [
        USAGE_COL,
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_holiday_wa",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "CDH",
        COL_MAX_T,
        COL_MIN_T,
        "temp_mean",
        "temp_range",
        "is_weekend_CDH",
        "is_night",
        "is_peak_5to9pm",
    ]

    xgboost_df = working[
        [
            c
            for c in features_to_keep
            + lag_cols
            + rolling_cols
            + weather_lags
            + rain_features
            + tmax_features
            + categorical_cols
            + ["y"]
            if c in working.columns
        ]
    ].copy()

    if meter_col and meter_col in working.columns:
        xgboost_df[meter_col] = working[meter_col]

    # Handle NaNs: forward/back fill per meter (if available)
    if meter_col:
        xgboost_df = xgboost_df.groupby(working[meter_col], group_keys=False).apply(lambda g: g.ffill().bfill())
    else:
        xgboost_df = xgboost_df.ffill().bfill()
    numeric_cols = xgboost_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        xgboost_df.loc[:, numeric_cols] = xgboost_df[numeric_cols].fillna(0.0)
    for col in [c for c in categorical_cols if c in xgboost_df.columns]:
        xgboost_df[col] = xgboost_df[col].astype("object").fillna("unknown")
    if meter_col and meter_col in xgboost_df.columns:
        xgboost_df[meter_col] = xgboost_df[meter_col].astype("object").fillna("unknown")

    xgboost_df = xgboost_df.dropna(subset=["y"])
    xgboost_df = xgboost_df.reset_index()
    if "index" in xgboost_df.columns and "timestamp" not in xgboost_df.columns:
        xgboost_df = xgboost_df.rename(columns={"index": "timestamp"})

    output_path = MODEL_DATA_DIR / "dataset_xgboost.csv"
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    xgboost_df.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)

    return DatasetArtefact(
        path=output_path,
        description="Tabular feature matrix for XGBoost with lags, rolls, and calendar encodings.",
        extra={"num_features": int(xgboost_df.shape[1] - 1), "num_rows": int(len(xgboost_df))},
    )


def compute_future_flag(series: pd.Series, horizon: int) -> pd.Series:
    shifted = series.shift(-1)
    future_max = (
        shifted[::-1]
        .rolling(window=horizon, min_periods=1)
        .max()[::-1]
    )
    return future_max.fillna(0).astype(int)


def prepare_xgboost_minimal_dataset(
    df: pd.DataFrame,
    meter_col: Optional[str],
) -> DatasetArtefact:
    minimal = pd.DataFrame(index=df.index.copy())
    minimal[USAGE_COL] = df[USAGE_COL]
    minimal["hour"] = df["hour"] if "hour" in df.columns else df.index.hour

    if meter_col:
        minimal[meter_col] = df[meter_col]
        minimal["y"] = df.groupby(meter_col)[USAGE_COL].shift(-1)
    else:
        minimal["y"] = df[USAGE_COL].shift(-1)

    minimal = minimal.dropna(subset=["y"])
    minimal = minimal.reset_index().rename(columns={"index": "timestamp"})

    ordered_cols: List[str] = ["timestamp"]
    if meter_col and meter_col in minimal.columns:
        ordered_cols.append(meter_col)
    ordered_cols.extend([USAGE_COL, "y", "hour"])
    minimal = minimal[ordered_cols]

    output_path = MODEL_DATA_DIR / "dataset_xgboost_minimal.csv"
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    minimal.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)

    return DatasetArtefact(
        path=output_path,
        description="Minimal XGBoost dataset with usage, hour, and next-hour target.",
        extra={"num_rows": int(len(minimal)), "columns": ordered_cols},
    )


def prepare_xgboost_disconnect_dataset(
    df: pd.DataFrame,
    meter_col: Optional[str],
    horizon: int = 24,
) -> DatasetArtefact:
    working = df.copy()
    if meter_col and meter_col not in working.columns:
        meter_col = None

    if "power_zero" not in working.columns:
        raise ValueError("Expected 'power_zero' column to build disconnect target.")

    if meter_col:
        future_flag = working.groupby(meter_col)["power_zero"].transform(
            lambda s: compute_future_flag(s, horizon)
        )
    else:
        future_flag = compute_future_flag(working["power_zero"], horizon)

    working["disconnect_within_24h"] = future_flag

    lag_cols = add_group_lags(
        working,
        USAGE_COL,
        lags=[1, 2, 24, 48, 168],
        group_key=meter_col,
        prefix="usage",
    )

    rolling_specs = {
        "roll_mean_3h": (3, "mean"),
        "roll_std_24h": (24, "std"),
        "roll_sum_168h": (168, "sum"),
    }
    rolling_cols = add_group_rolling(
        working,
        USAGE_COL,
        {label: (win, func) for label, (win, func) in rolling_specs.items()},
        meter_col,
    )

    weather_lags: List[str] = []
    rain_features: List[str] = []
    tmax_features: List[str] = []

    if COL_MAX_T in df.columns:
        weather_lags.extend(add_group_lags(working, COL_MAX_T, [24], meter_col, prefix="tmax"))
        tmax_roll_cols = add_group_rolling(
            working,
            COL_MAX_T,
            {"tmax_mean_168h": (168, "mean")},
            meter_col,
        )
        tmax_prev_cols: List[str] = []
        for col in tmax_roll_cols:
            prev_col = f"{col}_prevday"
            if meter_col:
                working[prev_col] = working.groupby(meter_col)[col].shift(24)
            else:
                working[prev_col] = working[col].shift(24)
            tmax_prev_cols.append(prev_col)
        working = working.drop(columns=tmax_roll_cols, errors="ignore")
        tmax_features.extend(tmax_prev_cols)

    if COL_SOLAR in df.columns:
        weather_lags.extend(add_group_lags(working, COL_SOLAR, [24], meter_col, prefix="solar"))

    if COL_RAIN in df.columns:
        rain_features.extend(add_group_lags(working, COL_RAIN, [24, 72, 168], meter_col, prefix="rain"))
        rain_roll_cols = add_group_rolling(
            working,
            COL_RAIN,
            {"rain_cum_72h": (72, "sum"), "rain_cum_168h": (168, "sum")},
            meter_col,
        )
        rain_prev_cols: List[str] = []
        for col in rain_roll_cols:
            prev_col = f"{col}_prevday"
            if meter_col:
                working[prev_col] = working.groupby(meter_col)[col].shift(24)
            else:
                working[prev_col] = working[col].shift(24)
            rain_prev_cols.append(prev_col)
        working = working.drop(columns=rain_roll_cols, errors="ignore")
        rain_features.extend(rain_prev_cols)

    categorical_cols: List[str] = []
    if meter_col:
        working["meter_code"] = working[meter_col].astype("category")
        categorical_cols.append("meter_code")

    features_to_keep = [
        USAGE_COL,
        "power_zero",
        "daily_energy_zero",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_holiday_wa",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_night",
        "is_peak_5to9pm",
        "CDH",
        COL_MAX_T,
        COL_MIN_T,
        "temp_mean",
        "temp_range",
    ]

    columns = [
        c
        for c in features_to_keep
        + lag_cols
        + rolling_cols
        + weather_lags
        + rain_features
        + tmax_features
        + categorical_cols
        + ["disconnect_within_24h"]
        if c in working.columns
    ]

    xgb_disc = working[columns].copy()
    if meter_col and meter_col in working.columns:
        xgb_disc[meter_col] = working[meter_col]

    if meter_col and meter_col in xgb_disc.columns:
        xgb_disc = xgb_disc.groupby(xgb_disc[meter_col], group_keys=False).apply(lambda g: g.ffill().bfill())
    else:
        xgb_disc = xgb_disc.ffill().bfill()

    numeric_cols = xgb_disc.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        xgb_disc.loc[:, numeric_cols] = xgb_disc[numeric_cols].fillna(0.0)

    for col in categorical_cols:
        if col in xgb_disc.columns:
            xgb_disc[col] = xgb_disc[col].astype("object").fillna("unknown")
    if meter_col and meter_col in xgb_disc.columns:
        xgb_disc[meter_col] = xgb_disc[meter_col].astype("object").fillna("unknown")

    xgb_disc = xgb_disc.dropna(subset=["disconnect_within_24h"])
    xgb_disc["disconnect_within_24h"] = xgb_disc["disconnect_within_24h"].astype(int)

    xgb_disc = xgb_disc.reset_index()
    if "index" in xgb_disc.columns and "timestamp" not in xgb_disc.columns:
        xgb_disc = xgb_disc.rename(columns={"index": "timestamp"})

    output_path = MODEL_DATA_DIR / "dataset_xgboost_disconnect.csv"
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    xgb_disc.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)

    return DatasetArtefact(
        path=output_path,
        description=f"Classification dataset for predicting disconnection within {horizon} hours.",
        extra={
            "num_features": int(xgb_disc.shape[1] - 1),
            "num_rows": int(len(xgb_disc)),
        },
    )


def build_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    meter_col: Optional[str],
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[pd.Timestamp]]:
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    meters: List[str] = []
    timestamps: List[pd.Timestamp] = []

    groups: Iterable[Tuple[str, pd.DataFrame]]
    if meter_col:
        groups = df.groupby(meter_col)
    else:
        groups = [(None, df)]

    for meter, group in groups:
        ordered = group.sort_index()
        features = ordered[feature_cols].values.astype(np.float32)
        target = ordered[target_col].shift(-1).values.astype(np.float32)
        index = ordered.index.to_list()

        for idx in range(lookback, len(ordered) - 1):
            window = features[idx - lookback: idx]
            y_value = target[idx]
            if np.isnan(y_value) or np.isnan(window).any():
                continue
            X_list.append(window)
            y_list.append(y_value)
            meters.append(str(meter) if meter is not None else "aggregate")
            timestamps.append(index[idx])

    if not X_list:
        raise ValueError("No LSTM sequences were generated; check that the dataset has enough rows.")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y, meters, timestamps


def prepare_lstm_dataset(
    df: pd.DataFrame,
    meter_col: Optional[str],
    lookback: int = 24,
) -> DatasetArtefact:
    feature_cols = [
        USAGE_COL,
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        "is_weekend", "is_holiday_wa",
        "CDH", "HDH",
        "temp_mean", "temp_range",
        COL_MAX_T, COL_MIN_T, COL_RAIN, COL_SOLAR,
        "rain_log1p", "solar_log1p",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    scalers: Dict[str, Dict[str, object]] = {}
    scaled_frames: List[pd.DataFrame] = []

    groups: Iterable[Tuple[str, pd.DataFrame]]
    if meter_col:
        groups = df.groupby(meter_col)
    else:
        groups = [(None, df)]

    for meter, group in groups:
        scaler = StandardScaler()
        values = group[feature_cols].astype(float)
        scaler.fit(values)
        scaled_values = scaler.transform(values)
        scaled_group = group.copy()
        scaled_group[feature_cols] = scaled_values
        scaled_frames.append(scaled_group)
        key = str(meter) if meter is not None else "aggregate"
        scalers[key] = {
            "feature_order": feature_cols,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }

    scaled_df = pd.concat(scaled_frames, axis=0).sort_index()

    X_seq, y_seq, meter_labels, target_times = build_lstm_sequences(
        scaled_df,
        feature_cols=feature_cols,
        target_col=USAGE_COL,
        meter_col=meter_col,
        lookback=lookback,
    )

    output_path = MODEL_DATA_DIR / "dataset_lstm.npz"
    np.savez_compressed(
        output_path,
        X=X_seq,
        y=y_seq,
        meter=np.asarray(meter_labels),
        timestamps=np.asarray(target_times, dtype="datetime64[ns]"),
        lookback=np.asarray([lookback]),
        feature_order=np.asarray(feature_cols),
    )

    scaler_path = MODEL_DATA_DIR / "lstm_scalers.json"
    with scaler_path.open("w", encoding="utf-8") as f:
        json.dump(scalers, f, indent=2)

    return DatasetArtefact(
        path=output_path,
        description=f"LSTM sequences (lookback {lookback} hours) with per-meter standardisation.",
        extra={
            "num_sequences": int(X_seq.shape[0]),
            "sequence_shape": list(X_seq.shape[1:]),
            "scalers": str(scaler_path),
        },
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-specific datasets.")
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["all"],
        help="List of model types to build (sarima, sarimax, xgboost, xgboost_disconnect, xgboost_minimal, lstm) or 'all'.",
    )
    parser.add_argument(
        "--lstm-lookback",
        type=int,
        default=24,
        help="Sliding window length (in hours) when generating LSTM sequences.",
    )
    return parser.parse_args()


def normalise_model_list(values: Sequence[str]) -> List[ModelType]:
    if not values:
        return list(ModelType)
    if any(val.lower() == "all" for val in values):
        return list(ModelType)
    return [ModelType.from_arg(val) for val in values]


def main() -> None:
    args = parse_args()
    requested_models = normalise_model_list(args.models)

    ensure_output_dir(MODEL_DATA_DIR)

    df, meter_col = prepare_base_dataframe()
    artefacts: List[DatasetArtefact] = []
    dropped_features_report: Dict[str, List[str]] = {}

    for model in requested_models:
        if model is ModelType.SARIMA:
            artefacts.append(prepare_sarima_dataset(df, meter_col))
            dropped_features_report[model.value] = []
        elif model is ModelType.SARIMAX:
            artefact = prepare_sarimax_dataset(df, meter_col)
            artefacts.append(artefact)
            dropped_features_report[model.value] = artefact.extra.get("dropped_multicollinear", []) if artefact.extra else []
        elif model is ModelType.XGBOOST:
            artefacts.append(prepare_xgboost_dataset(df, meter_col))
            dropped_features_report[model.value] = []
        elif model is ModelType.XGBOOST_MINIMAL:
            artefacts.append(prepare_xgboost_minimal_dataset(df, meter_col))
            dropped_features_report[model.value] = []
        elif model is ModelType.LSTM:
            artefacts.append(prepare_lstm_dataset(df, meter_col, lookback=args.lstm_lookback))
            dropped_features_report[model.value] = []

    print("Datasets generated:")
    for artefact in artefacts:
        print(f" - {artefact.path}: {artefact.description}")
        if artefact.extra:
            for key, value in artefact.extra.items():
                print(f"    {key}: {value}")

    print("\nDropped multicollinear feature summary:")
    for model_name, features in dropped_features_report.items():
        if features:
            print(f" - {model_name}: {features}")
        else:
            print(f" - {model_name}: none")


if __name__ == "__main__":
    main()
