"""Minimal SARIMA trainer for aggregate and per-meter workloads with logging.

This script loads the feature-engineered time-series dataset and trains SARIMA
models using a straightforward grid-search. Outputs (predictions, metrics, run
logs) are written under models/sarima/ within a unique run folder so that each
execution remains self-contained.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:  # Support running both as a module and as a script
    from .output_utils import prepare_run_directory
except ImportError:  
    from output_utils import prepare_run_directory


DEFAULT_DATA_PATH = Path("model_datasets/dataset_timeseries.csv")
DEFAULT_TIMESTAMP = "timestamp"
DEFAULT_TARGET = "delivered_value"
DEFAULT_METER_COL = "meter_ui"
DEFAULT_OUTPUT_DIR = Path("models/sarima")

DEFAULT_P = [0, 1, 2]
DEFAULT_D = [1]
DEFAULT_Q = [0, 1, 2]
DEFAULT_SP = [0, 1, 2]
DEFAULT_SD = [0, 1]
DEFAULT_SQ = [0, 1, 2]
DEFAULT_SEASONAL_PERIOD = 24


@dataclass
class SeriesResult:
    name: str
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    metrics: Dict[str, float]
    predictions_path: Path
    metrics_path: Path
    tried: int
    failed: int
    train_size: int
    val_size: int


def smape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1e-9
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape(y_true, y_pred)}


def parse_int_list(values: Optional[Sequence[int]], default: Sequence[int]) -> List[int]:
    if values is None or len(values) == 0:
        return list(default)
    return [int(v) for v in values]


def load_dataset(path: Path, timestamp_col: str, target_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column '{timestamp_col}' in dataset")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' in dataset")

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col, target_col])
    df = df.sort_values(timestamp_col)
    return df


def detect_meter_column(df: pd.DataFrame, preferred: str = DEFAULT_METER_COL) -> Optional[str]:
    if preferred in df.columns:
        return preferred
    for candidate in ("meter_ui", "nmi_ui"):
        if candidate in df.columns:
            return candidate
    return None


def prepare_series(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    meter_col: Optional[str],
    meter_value: Optional[str],
) -> Optional[pd.Series]:
    if meter_col and meter_value is not None:
        subset = df[df[meter_col] == meter_value].copy()
        if subset.empty:
            return None
    else:
        subset = df.copy()

    series = subset.set_index(timestamp_col)[target_col].dropna()
    series = series.astype(float)
    return series.sort_index() if not series.empty else None


def aggregate_series(df: pd.DataFrame, timestamp_col: str, target_col: str) -> pd.Series:
    aggregated = df.groupby(timestamp_col, as_index=True)[target_col].sum().astype(float)
    return aggregated.sort_index()


def build_orders(p_vals: Sequence[int], d_vals: Sequence[int], q_vals: Sequence[int]) -> List[Tuple[int, int, int]]:
    return [(p, d, q) for p in p_vals for d in d_vals for q in q_vals]


def build_seasonal_orders(
    P_vals: Sequence[int],
    D_vals: Sequence[int],
    Q_vals: Sequence[int],
    seasonal_period: int,
) -> List[Tuple[int, int, int, int]]:
    return [(P, D, Q, seasonal_period) for P in P_vals for D in D_vals for Q in Q_vals]


def split_train_val(series: pd.Series, val_hours: int) -> Tuple[pd.Series, pd.Series]:
    if val_hours <= 0 or val_hours >= len(series):
        raise ValueError("val_hours must be positive and smaller than the series length")
    train = series.iloc[:-val_hours]
    val = series.iloc[-val_hours:]
    return train, val


def fit_sarima(
    train: pd.Series,
    order: Tuple[int, int, int],
    seasonal: Tuple[int, int, int, int],
    maxiter: int,
) -> Optional[SARIMAX]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal,
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            return model.fit(disp=False, maxiter=maxiter)
    except Exception:
        return None


def grid_search(
    train: pd.Series,
    val: pd.Series,
    orders: Sequence[Tuple[int, int, int]],
    seasonal_orders: Sequence[Tuple[int, int, int, int]],
    maxiter: int,
) -> Tuple[Optional[SARIMAX], Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int, int]], Optional[np.ndarray], int, int]:
    best_model = None
    best_order = None
    best_seasonal = None
    best_pred = None
    best_rmse = np.inf
    tried = 0
    failed = 0

    total_combos = len(orders) * len(seasonal_orders)
    logger = logging.getLogger('sarima_trainer')
    for idx_o, order in enumerate(orders, start=1):
        for idx_s, seasonal in enumerate(seasonal_orders, start=1):
            combo_index = (idx_o - 1) * len(seasonal_orders) + idx_s
            logger.debug("Grid search (%s/%s): order=%s seasonal=%s", combo_index, total_combos, order, seasonal)
            result = fit_sarima(train, order, seasonal, maxiter)
            if result is None:
                failed += 1
                continue
            try:
                pred = result.forecast(steps=len(val))
            except Exception:
                failed += 1
                continue
            try:
                rmse = mean_squared_error(val, pred, squared=False)
            except TypeError:
                rmse = mean_squared_error(val, pred) ** 0.5
            if not np.isfinite(rmse):
                failed += 1
                continue
            tried += 1
            logger.debug("Result (%s/%s): order=%s seasonal=%s RMSE=%.4f", combo_index, total_combos, order, seasonal, rmse)
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = result
                best_order = order
                best_seasonal = seasonal
                best_pred = pred
    return best_model, best_order, best_seasonal, best_pred, tried, failed


def run_series(
    name: str,
    series: pd.Series,
    val_hours: int,
    orders: Sequence[Tuple[int, int, int]],
    seasonal_orders: Sequence[Tuple[int, int, int, int]],
    maxiter: int,
    output_dir: Path,
    logger: logging.Logger,
) -> Optional[SeriesResult]:
    if len(series) <= val_hours:
        logger.warning(
            "[%s] Skipping: series shorter than validation window (%s <= %s)",
            name,
            len(series),
            val_hours,
        )
        return None

    train, val = split_train_val(series, val_hours)
    model, order, seasonal, pred, tried, failed = grid_search(train, val, orders, seasonal_orders, maxiter)
    if model is None or order is None or seasonal is None or pred is None:
        logger.warning("[%s] No valid model found after testing %s combinations", name, tried + failed)
        return None

    metrics = compute_metrics(val, pred)

    series_dir = output_dir / name
    series_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = series_dir / "predictions.csv"
    metrics_path = series_dir / "metrics.json"

    predictions_df = pd.DataFrame(
        {
            "timestamp": val.index,
            "y_true": val.values,
            "y_pred": pred,
        }
    )
    predictions_df.to_csv(predictions_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "order": order,
                "seasonal_order": seasonal,
                "metrics": metrics,
                "tried": tried,
                "failed": failed,
                "val_hours": val_hours,
                "train_size": len(train),
                "val_size": len(val),
            },
            f,
            indent=2,
        )

    logger.info(
        "[%s] Best order=%s, seasonal=%s, RMSE=%.4f (tried %s, failed %s)",
        name,
        order,
        seasonal,
        metrics["RMSE"],
        tried,
        failed,
    )

    return SeriesResult(
        name=name,
        order=order,
        seasonal_order=seasonal,
        metrics=metrics,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        tried=tried,
        failed=failed,
        train_size=len(train),
        val_size=len(val),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast SARIMA trainer with simple grid-search.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to dataset_timeseries.csv")
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP, help="Timestamp column name")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column name")
    parser.add_argument("--meter-column", default=DEFAULT_METER_COL, help="Meter identifier column")
    parser.add_argument("--meter", default="ALL", help="Meter to train. Use 'ALL' for aggregate or combine with --per-meter")
    parser.add_argument("--per-meter", action="store_true", help="Train a model for every meter in the dataset")
    parser.add_argument("--val-hours", type=int, default=7 * 24, help="Validation window (hours)")
    parser.add_argument("--seasonal-period", type=int, default=DEFAULT_SEASONAL_PERIOD, help="Seasonal period")
    parser.add_argument("--p", type=int, nargs="*", help="AR p values to consider")
    parser.add_argument("--d", type=int, nargs="*", help="Differencing d values to consider")
    parser.add_argument("--q", type=int, nargs="*", help="MA q values to consider")
    parser.add_argument("--P", type=int, nargs="*", help="Seasonal AR P values")
    parser.add_argument("--D", type=int, nargs="*", help="Seasonal differencing D values")
    parser.add_argument("--Q", type=int, nargs="*", help="Seasonal MA Q values")
    parser.add_argument("--maxiter", type=int, default=100, help="Max iterations for SARIMAX fitting")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Base directory for outputs")
    parser.add_argument("--run-name", help="Optional run name (subdirectory)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    parser.add_argument("--no-timestamp", action="store_true", help="Do not append a timestamp to the run directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = logging.getLogger("sarima_trainer")
    logger.setLevel(log_level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    try:
        p_vals = parse_int_list(args.p, DEFAULT_P)
        d_vals = parse_int_list(args.d, DEFAULT_D)
        q_vals = parse_int_list(args.q, DEFAULT_Q)
        P_vals = parse_int_list(args.P, DEFAULT_SP)
        D_vals = parse_int_list(args.D, DEFAULT_SD)
        Q_vals = parse_int_list(args.Q, DEFAULT_SQ)

        if not all((p_vals, d_vals, q_vals, P_vals, D_vals, Q_vals)):
            raise ValueError("At least one parameter grid list is empty; provide values for p/d/q and P/D/Q.")

        orders = build_orders(p_vals, d_vals, q_vals)
        seasonal_orders = build_seasonal_orders(P_vals, D_vals, Q_vals, args.seasonal_period)
        if not orders or not seasonal_orders:
            raise ValueError("Parameter grids produced zero combinations; adjust your p/d/q or P/D/Q values.")

        run_dir = prepare_run_directory(args.output_dir, args.run_name, timestamp=not args.no_timestamp)

        file_handler = logging.FileHandler(run_dir / "run.log", mode="w", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_handler)

        logger.info("Writing outputs to %s", run_dir)

        df = load_dataset(args.data_path, args.timestamp, args.target)
        meter_col = detect_meter_column(df, args.meter_column)

        def serialise_result(res: SeriesResult) -> Dict[str, object]:
            def _rel(path: Path) -> str:
                try:
                    return str(path.relative_to(run_dir))
                except ValueError:
                    return str(path)

            return {
                "order": res.order,
                "seasonal_order": res.seasonal_order,
                "metrics": res.metrics,
                "tried": res.tried,
                "failed": res.failed,
                "train_size": res.train_size,
                "val_size": res.val_size,
                "predictions_path": _rel(res.predictions_path),
                "metrics_path": _rel(res.metrics_path),
            }

        if args.per_meter:
            if meter_col is None:
                raise ValueError("Meter column not found; cannot train per-meter models.")
            meters = sorted(df[meter_col].dropna().unique())
            if not meters:
                raise ValueError("No meters available for training in the dataset.")
            logger.info("Training %s meters (per_meter mode).", len(meters))
            results: Dict[str, Dict[str, object]] = {}
            for meter in meters:
                series = prepare_series(df, args.timestamp, args.target, meter_col, meter)
                if series is None:
                    logger.warning("[%s] Skipping: no data after filtering.", meter)
                    continue
                res = run_series(str(meter), series, args.val_hours, orders, seasonal_orders, args.maxiter, run_dir, logger)
                if res:
                    results[str(meter)] = serialise_result(res)
            if results:
                summary_path = run_dir / "summary.json"
                with summary_path.open("w", encoding="utf-8") as f:
                    json.dump({"mode": "per_meter", "results": results}, f, indent=2)
                logger.info("Per-meter summary written to %s", summary_path)
            else:
                logger.warning("No meters produced successful models.")
            return

        if args.meter.upper() == "ALL":
            logger.info("Training aggregate model across all meters.")
            series = aggregate_series(df, args.timestamp, args.target)
            res = run_series("aggregate", series, args.val_hours, orders, seasonal_orders, args.maxiter, run_dir, logger)
            if res:
                summary_path = run_dir / "summary.json"
                with summary_path.open("w", encoding="utf-8") as f:
                    json.dump({"mode": "aggregate", "result": serialise_result(res)}, f, indent=2)
                logger.info("Aggregate summary written to %s", summary_path)
            return

        if meter_col is None:
            raise ValueError("Meter column not found; cannot select a specific meter.")

        series = prepare_series(df, args.timestamp, args.target, meter_col, args.meter)
        if series is None:
            raise ValueError(f"Meter '{args.meter}' not found or has no data")
        logger.info("Training meter %s", args.meter)
        res = run_series(str(args.meter), series, args.val_hours, orders, seasonal_orders, args.maxiter, run_dir, logger)
        if res:
            summary_path = run_dir / "summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump({"mode": "single", "result": serialise_result(res)}, f, indent=2)
            logger.info("Summary written to %s", summary_path)

    except Exception as exc:
        logger.exception("SARIMA run failed: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    np.seterr(all="ignore")
    logging.captureWarnings(True)
    main()
