"""Prophet training script supporting aggregate, single-meter, and per-meter forecasts.

Usage examples
--------------
python modelling/prophet.py --val_hours 168 --forecast_horizon 6 --run_name baseline
python modelling/prophet.py --mode single --meter_column meter_ui --meter M1 --extra_regressors temp_mean is_holiday_wa --run_name m1
python modelling/prophet.py --mode per_meter --meter_column meter_ui --output_dir models/prophet --run_name per_meter

The script:
- loads `model_datasets/dataset_timeseries.csv` by default
- validates and aggregates features by timestamp (summing the target, averaging regressors)
- trains Prophet either on the global aggregate, a single meter, or every meter
- evaluates on the final `val_hours` window
- writes forecasts and metrics to a per-run directory under the chosen output base (defaults to `models/prophet`)

Extra regressors can be supplied with `--extra_regressors col1 col2`; remember to extend their future values when forecasting beyond the observed history.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from pandas.errors import MergeError
import numpy as np
import pandas as pd

try:
    from .output_utils import prepare_run_directory
except ImportError:  # pragma: no cover - fallback when run as script VERY occasionally fails 
    from output_utils import prepare_run_directory

# Work around local filename shadowing the installed prophet package.(error from earlier, fixed now, but keep this here just in case)
_SCRIPT_DIR = Path(__file__).resolve().parent
_removed = False
if str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))
    _removed = True
try:
    Prophet = importlib.import_module("prophet").Prophet
finally:
    if _removed:
        sys.path.insert(0, str(_SCRIPT_DIR))
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_FILE = "model_datasets/dataset_timeseries.csv"
TARGET = "delivered_value"
TIMESTAMP = "timestamp"
DEFAULT_OUTPUT_DIR = Path("models/prophet")
DEFAULT_METER_COL = "meter_ui"


def smape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1e-9
    return 100 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)


def evaluate(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape(y_true, y_pred)}


def load_dataset(
    path: Path,
    timestamp_col: str,
    target_col: str,
    extra_regressors: List[str],
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Dataset must include timestamp column '{timestamp_col}'.")
    if target_col not in df.columns:
        raise ValueError(f"Dataset must include target column '{target_col}'.")
    missing_regressors = [reg for reg in extra_regressors if reg not in df.columns]
    if missing_regressors:
        raise ValueError(f"Extra regressors not found: {missing_regressors}")

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col)
    return df


def prepare_series(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    extra_regressors: List[str],
    aggregate: bool,
    meter_col: Optional[str] = None,
    meter_value: Optional[str] = None,
) -> pd.DataFrame:
    if meter_col is not None and meter_value is not None:
        working = df[df[meter_col] == meter_value].copy()
        if working.empty:
            raise ValueError(f"No rows found for {meter_col}='{meter_value}'.")
    else:
        working = df.copy()

    agg_spec: Dict[str, str] = {target_col: "sum"}
    for reg in extra_regressors:
        agg_spec[reg] = "mean"

    grouped = (
        working.groupby(timestamp_col, as_index=False)[list(agg_spec.keys())]
        .agg(agg_spec)
        .sort_values(timestamp_col)
    )
    grouped.rename(columns={timestamp_col: "ds", target_col: "y"}, inplace=True)
    grouped = grouped.dropna(subset=["ds", "y"])

    if aggregate and meter_col is not None:
        grouped.insert(0, meter_col, meter_value if meter_value is not None else "ALL")

    return grouped


def split_train_val(df: pd.DataFrame, val_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if val_hours <= 0:
        raise ValueError("val_hours must be positive.")
    if val_hours >= len(df):
        raise ValueError("val_hours must be smaller than the length of the series.")
    return df.iloc[:-val_hours].copy(), df.iloc[-val_hours:].copy()


def prepare_future_dataframe(
    model: Prophet,
    history: pd.DataFrame,
    freq: str,
    val_hours: int,
    forecast_horizon: int,
    extra_regressors: List[str],
) -> pd.DataFrame:
    freq_str = str(freq).lower()
    periods = max(0, val_hours) + max(0, forecast_horizon)
    future = model.make_future_dataframe(periods=periods, freq=freq_str, include_history=True)
    if extra_regressors:
        future = future.merge(history[["ds"] + extra_regressors], on="ds", how="left")
        missing = future[extra_regressors].isna().any(axis=1)
        if missing.any():
            raise ValueError(
                "Future dataframe has missing regressor values. Provide regressor values for future periods."
            )
    return future


def fit_and_forecast(
    series_df: pd.DataFrame,
    extra_regressors: List[str],
    val_hours: int,
    forecast_horizon: int,
    freq: str,
    changepoint_prior_scale: float,
    seasonality_mode: str,
    growth: str,
    weekly_seasonality: str,
    daily_seasonality: str,
    series_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if len(series_df) <= val_hours:
        raise ValueError("Not enough samples to hold out the requested validation window.")

    train_df, val_df = split_train_val(series_df, val_hours)

    model = Prophet(
        growth=growth,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
    )
    for reg in extra_regressors:
        model.add_regressor(reg)

    model.fit(train_df)

    future = prepare_future_dataframe(model, series_df, freq, val_hours, forecast_horizon, extra_regressors)
    forecast = model.predict(future)

    joined = forecast.merge(series_df[["ds", "y"]], on="ds", how="left")
    try:
        val_forecast = val_df[["ds", "y"]].merge(
            forecast[["ds", "yhat"]],
            on="ds",
            how="left",
            validate="one_to_one",
        )
    except MergeError as exc:
        raise RuntimeError(
            "Validation alignment mismatch; consider adjusting --freq or cleaning duplicate timestamps."
        ) from exc
    missing_mask = val_forecast["yhat"].isna()
    if missing_mask.any():
        raise RuntimeError(
            "Missing Prophet predictions for some validation timestamps; ensure your data covers the chosen frequency or provide matching future regressor values."
        )

    metrics = evaluate(val_forecast["y"], val_forecast["yhat"])

    if series_label is not None and "series_id" not in joined.columns:
        joined = joined.copy()
        joined.insert(0, "series_id", series_label)

    return joined, metrics


def save_outputs(
    forecast: pd.DataFrame,
    metrics: Dict[str, float],
    output_dir: Path,
    series_label: Optional[str] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = output_dir / "forecast.csv"
    metrics_path = output_dir / "metrics.json"
    to_save = forecast.copy()
    if series_label is not None and "series_id" not in to_save.columns:
        to_save.insert(0, "series_id", series_label)
    to_save.to_csv(forecast_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def slugify(value: object) -> str:
    text = str(value)
    safe_chars = [c if c.isalnum() or c in ("-", "_") else "_" for c in text]
    slug = "".join(safe_chars).strip("_")
    return slug or "series"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Prophet on aggregate or per-meter time series.")
    parser.add_argument("--data_path", type=Path, default=Path(DATA_FILE), help="Path to the input CSV data.")
    parser.add_argument("--timestamp", default=TIMESTAMP, help="Timestamp column name in the dataset.")
    parser.add_argument("--target", default=TARGET, help="Target column name to forecast.")
    parser.add_argument("--mode", choices=["aggregate", "single", "per_meter"], default="aggregate", help="Training mode: aggregate across meters, a single meter, or loop over every meter.")
    parser.add_argument("--meter_column", default=DEFAULT_METER_COL, help="Column containing the meter identifier.")
    parser.add_argument("--meter", help="Meter identifier to train when --mode single.")
    parser.add_argument("--max_meters", type=int, help="Optional cap on the number of meters to process in per_meter mode.")
    parser.add_argument("--val_hours", type=int, default=168, help="Number of trailing hours for validation.")
    parser.add_argument("--forecast_horizon", type=int, default=6, help="Number of future periods (hours) to forecast beyond the historical data.")
    parser.add_argument("--freq", default="h", help="Pandas frequency string for the time series (default hourly).")
    parser.add_argument("--extra_regressors", nargs="*", default=[], help="Optional extra regressor column names.")
    parser.add_argument("--changepoint_prior_scale", type=float, default=0.05, help="Controls trend flexibility; lower values keep the trend smoother.")
    parser.add_argument("--seasonality_mode", choices=["additive", "multiplicative"], default="additive", help="Seasonality mode for Prophet.")
    parser.add_argument("--growth", choices=["linear", "logistic"], default="linear", help="Trend growth type.")
    parser.add_argument("--weekly_seasonality", default="auto", help="Weekly seasonality setting passed to Prophet (True/False/auto).")
    parser.add_argument("--daily_seasonality", default="auto", help="Daily seasonality setting passed to Prophet (True/False/auto).")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Base directory to store forecast outputs.")
    parser.add_argument("--run_name", help="Optional name for this run; becomes the subdirectory under the output base.")
    parser.add_argument("--no_timestamp", action="store_true", help="Do not append an automatic timestamp to the run directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = prepare_run_directory(
        args.output_dir,
        args.run_name,
        timestamp=not args.no_timestamp,
    )
    args.output_dir = run_dir
    print(f"Outputs will be written to {run_dir}")

    df = load_dataset(
        path=args.data_path,
        timestamp_col=args.timestamp,
        target_col=args.target,
        extra_regressors=args.extra_regressors,
    )

    if args.mode == "aggregate":
        series = prepare_series(
            df,
            timestamp_col=args.timestamp,
            target_col=args.target,
            extra_regressors=args.extra_regressors,
            aggregate=True,
        )
        forecast, metrics = fit_and_forecast(
            series_df=series,
            extra_regressors=args.extra_regressors,
            val_hours=args.val_hours,
            forecast_horizon=args.forecast_horizon,
            freq=args.freq,
            changepoint_prior_scale=args.changepoint_prior_scale,
            seasonality_mode=args.seasonality_mode,
            growth=args.growth,
            weekly_seasonality=args.weekly_seasonality,
            daily_seasonality=args.daily_seasonality,
            series_label="aggregate",
        )
        save_outputs(forecast, metrics, args.output_dir, series_label="aggregate")
        print(f"Saved aggregate forecast to {args.output_dir / 'forecast.csv'}")
        print(f"Saved aggregate metrics to {args.output_dir / 'metrics.json'}")
        return

    meter_col = args.meter_column
    if meter_col not in df.columns:
        raise ValueError(f"Meter column '{meter_col}' not found in dataset.")

    if args.mode == "single":
        if not args.meter:
            raise ValueError("--meter must be provided when --mode single.")
        series = prepare_series(
            df,
            timestamp_col=args.timestamp,
            target_col=args.target,
            extra_regressors=args.extra_regressors,
            aggregate=False,
            meter_col=meter_col,
            meter_value=args.meter,
        )
        label = str(args.meter)
        forecast, metrics = fit_and_forecast(
            series_df=series,
            extra_regressors=args.extra_regressors,
            val_hours=args.val_hours,
            forecast_horizon=args.forecast_horizon,
            freq=args.freq,
            changepoint_prior_scale=args.changepoint_prior_scale,
            seasonality_mode=args.seasonality_mode,
            growth=args.growth,
            weekly_seasonality=args.weekly_seasonality,
            daily_seasonality=args.daily_seasonality,
            series_label=label,
        )
        meter_dir = args.output_dir / f"{meter_col}_{slugify(label)}"
        save_outputs(forecast, metrics, meter_dir, series_label=label)
        print(f"Saved forecast to {meter_dir / 'forecast.csv'}")
        print(f"Saved metrics to {meter_dir / 'metrics.json'}")
        return

    # per_meter mode
    meters = df[meter_col].dropna().unique()
    meters = sorted(meters)
    if args.max_meters is not None:
        meters = meters[: args.max_meters]
    if not meters:
        raise ValueError("No meters found to train.")

    summary: Dict[str, Dict[str, float]] = {}
    for meter_value in meters:
        label = str(meter_value)
        try:
            series = prepare_series(
                df,
                timestamp_col=args.timestamp,
                target_col=args.target,
                extra_regressors=args.extra_regressors,
                aggregate=False,
                meter_col=meter_col,
                meter_value=meter_value,
            )
            forecast, metrics = fit_and_forecast(
                series_df=series,
                extra_regressors=args.extra_regressors,
                val_hours=args.val_hours,
                forecast_horizon=args.forecast_horizon,
                freq=args.freq,
                changepoint_prior_scale=args.changepoint_prior_scale,
                seasonality_mode=args.seasonality_mode,
                growth=args.growth,
                weekly_seasonality=args.weekly_seasonality,
                daily_seasonality=args.daily_seasonality,
                series_label=label,
            )
        except ValueError as exc:
            print(f"[{label}] Skipping: {exc}")
            continue
        meter_dir = args.output_dir / f"{meter_col}_{slugify(label)}"
        save_outputs(forecast, metrics, meter_dir, series_label=label)
        summary[label] = metrics
        print(f"[{label}] Saved outputs to {meter_dir}")

    if not summary:
        raise RuntimeError("No meters were successfully trained; see logs above.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote per-meter metrics summary to {summary_path}")


if __name__ == "__main__":
    main()
