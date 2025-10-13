# train_sarimax.py
# -------------------------------------------
# SARIMAX with grid search on hourly electricity data (per meter).
# Exogenous regressors: weather + calendar/cyclical features from dataset_timeseries.csv
#
# Usage:
#   python train_sarimax.py --meter M1 --val_hours 336 --save_model --run_name m1
#   python train_sarimax.py --meter ALL --val_hours 336 --run_name all
# -------------------------------------------
import argparse
import csv
import json
import time
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
np.seterr(all="ignore")

try:
    from .output_utils import prepare_run_directory
except ImportError:  # pragma: no cover - fallback when run as script
    from output_utils import prepare_run_directory


# -----------------------------
# Config
# -----------------------------
DATA_FILE = "model_datasets/dataset_timeseries.csv"
TARGET = "delivered_value"
TIMESTAMP = "timestamp"
ID_COLS_CANDIDATES = ["meter_ui", "nmi_ui"]

# Default grid (sensible + not insane)
PDQ = [(p, d, q) for p in [0, 1, 2] for d in [0, 1] for q in [0, 1, 2]]
SPQ = [(P, D, Q) for P in [0, 1, 2] for D in [0, 1] for Q in [0, 1, 2]]
COARSE_PDQ = PDQ[: min(8, len(PDQ))]
COARSE_SPQ = SPQ[: min(8, len(SPQ))]
NEIGHBOR_DELTAS = [(-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0)]
PDQ_SET = set(PDQ)
SPQ_SET = set(SPQ)
# Total 16 candidates instead of hundreds
SEASONAL_PERIOD = 24  # hourly data → daily seasonality

# Exogenous features we expect from the engineered dataset
EXOG_CANDIDATES = [
    # calendar & cyclical
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_holiday_wa", "is_night", "is_peak_5to9pm",
    "hour", "month", "dow",
    # weather & derivatives
    "maximum_temperature_degree_c", "minimum_temperature_degree_c",
    "rainfall_amount_millimetres", "daily_global_solar_exposure_mj_m_m",
    "temp_mean", "temp_range", "is_hot_day", "is_cold_night",
    "CDH", "HDH",
]

DEFAULT_OUTPUT_DIR = Path("models/sarimax")

# -----------------------------
# Metrics
# -----------------------------
def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1e-9
    return 100 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100
    s = smape(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": s}



def infinite_metrics():
    return {"MAE": np.inf, "RMSE": np.inf, "MAPE": np.inf, "sMAPE": np.inf}


class GridSearchProgressLogger:
    def __init__(self, path):
        self.path = Path(path)
        self._header_written = self.path.exists()

    def _ensure_header(self):
        if not self._header_written:
            if self.path.parent and not self.path.parent.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'meter', 'stage', 'index', 'total',
                    'order', 'seasonal_order', 'rmse', 'mae', 'mape', 'smape',
                    'elapsed_seconds', 'error'
                ])
            self._header_written = True

    def log(self, meter, stage, index, total, order, seasonal_order, metrics, elapsed, error_message=None):
        self._ensure_header()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        order_str = repr(order) if order is not None else ''
        seasonal_str = repr(seasonal_order) if seasonal_order is not None else ''
        clean_error = (error_message or '').replace('\n', ' ').replace('\r', ' ').strip()
        if clean_error and len(clean_error) > 240:
            clean_error = clean_error[:240] + '...'
        row = [
            timestamp,
            meter,
            stage,
            index,
            total,
            order_str,
            seasonal_str,
            metrics.get('RMSE', np.inf),
            metrics.get('MAE', np.inf),
            metrics.get('MAPE', np.inf),
            metrics.get('sMAPE', np.inf),
            f"{elapsed:.3f}",
            clean_error,
        ]
        with self.path.open('a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


# -----------------------------
# Data loading & prep
# -----------------------------
def load_data(path: Union[str, Path]):
    df = pd.read_csv(path)
    if TIMESTAMP not in df.columns:
        raise ValueError(f"{path} must include a '{TIMESTAMP}' column.")
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    # identify meter column (default to meter_ui if present)
    meter_col = None
    for c in ID_COLS_CANDIDATES:
        if c in df.columns:
            meter_col = c
            break
    return df, meter_col

def get_exog_cols(df: pd.DataFrame):
    """Use only the exogenous columns that are actually present & numeric (and not the target)."""
    present = [c for c in EXOG_CANDIDATES if c in df.columns]
    # keep numeric only
    present = [c for c in present if pd.api.types.is_numeric_dtype(df[c])]
    # drop ones with all NaN
    present = [c for c in present if df[c].notna().any()]
    # just to be safe, do not include target itself
    present = [c for c in present if c != TARGET]
    return present


def split_constant_exog(df: pd.DataFrame, exog_cols):
    if not exog_cols:
        return [], []
    keep = []
    dropped = []
    for col in exog_cols:
        series = df[col]
        if series.dropna().nunique() > 1:
            keep.append(col)
        else:
            dropped.append(col)
    return keep, dropped

def split_train_val(df: pd.DataFrame, val_hours: int):
    """Time-based split: last val_hours go to validation."""
    if val_hours <= 0 or val_hours >= len(df):
        raise ValueError("val_hours must be >0 and < len(series).")
    train = df.iloc[:-val_hours].copy()
    val = df.iloc[-val_hours:].copy()
    return train, val


# -----------------------------
# Grid search SARIMAX
# -----------------------------
def fit_and_forecast(train, val, exog_cols, order, sorder, seasonal_period, maxiter=200):
    """Fit SARIMAX on train, forecast len(val) using exog_val."""
    if order is None or sorder is None:
        return None, None, infinite_metrics(), 'Missing order or seasonal_order'

    if exog_cols:
        train_ = train[[TARGET] + exog_cols].dropna().copy()
        val_ = val[[TARGET] + exog_cols].dropna().copy()
        y_train = train_[TARGET]
        X_train = train_[exog_cols]
        y_val = val_[TARGET]
        X_val = val_[exog_cols]
    else:
        train_ = train[[TARGET]].dropna().copy()
        val_ = val[[TARGET]].dropna().copy()
        y_train = train_[TARGET]
        X_train = None
        y_val = val_[TARGET]
        X_val = None

    if train_.empty or val_.empty:
        return None, None, infinite_metrics(), 'Insufficient data after dropping missing values'

    if len(y_train) < max(10, seasonal_period * 2):
        return None, None, infinite_metrics(), 'Not enough history to fit seasonal model'

    try:
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=order,
            seasonal_order=(sorder[0], sorder[1], sorder[2], seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend='c'
        )
        res = model.fit(maxiter=maxiter, disp=False)
        fcst = res.forecast(steps=len(val_), exog=X_val)
        metrics = evaluate(y_val.values, fcst.values)
        return res, fcst, metrics, None
    except Exception as exc:
        return None, None, infinite_metrics(), str(exc)



def grid_search(train, val, exog_cols, seasonal_period, meter_tag=None, coarse_first=True, progress_logger=None):
    meter_label = meter_tag or "SARIMAX"
    total_tried = 0
    best_global = {"order": None, "sorder": None, "metrics": {"RMSE": np.inf}}
    seen_combos = set()

    def run_stage(pdq_list, spq_list, stage_name):
        nonlocal total_tried, best_global
        combos = [
            (order, sorder)
            for order in pdq_list
            for sorder in spq_list
            if order in PDQ_SET and sorder in SPQ_SET
        ]
        combos = [combo for combo in combos if combo not in seen_combos]
        if not combos:
            return {"order": None, "sorder": None, "metrics": {"RMSE": np.inf}}
        stage_best = {"order": None, "sorder": None, "metrics": {"RMSE": np.inf}}
        last_error = None
        total = len(combos)
        for idx, (order, sorder) in enumerate(combos, start=1):
            seen_combos.add((order, sorder))
            total_tried += 1
            if meter_tag is not None:
                print(f"[{meter_label}] {stage_name} {idx}/{total} order={order}, seasonal={sorder}", end="\r", flush=True)
            start_time = time.perf_counter()
            _, _, metrics, error_msg = fit_and_forecast(train, val, exog_cols, order, sorder, seasonal_period)
            elapsed = time.perf_counter() - start_time
            if progress_logger:
                progress_logger.log(
                    meter=meter_label,
                    stage=stage_name,
                    index=idx,
                    total=total,
                    order=order,
                    seasonal_order=sorder,
                    metrics=metrics,
                    elapsed=elapsed,
                    error_message=error_msg,
                )
            if error_msg:
                last_error = error_msg
            if metrics["RMSE"] < stage_best["metrics"]["RMSE"]:
                stage_best = {"order": order, "sorder": sorder, "metrics": metrics}
            if metrics["RMSE"] < best_global["metrics"]["RMSE"]:
                best_global = {"order": order, "sorder": sorder, "metrics": metrics}
        if meter_tag is not None:
            print(" " * 80, end="\r")
            if stage_best["order"] is not None:
                print(f"[{meter_label}] {stage_name} best order={stage_best['order']}, seasonal={stage_best['sorder']}, RMSE={stage_best['metrics']['RMSE']:.4f}")
            else:
                if last_error:
                    preview = last_error.replace('\n', ' ').replace('\r', ' ')
                    if len(preview) > 200:
                        preview = preview[:200] + '...'
                    print(f"[{meter_label}] {stage_name} no valid fits. Last error: {preview}")
                else:
                    print(f"[{meter_label}] {stage_name} no valid fits.")
        return stage_best

    if coarse_first and COARSE_PDQ and COARSE_SPQ:
        coarse_best = run_stage(COARSE_PDQ, COARSE_SPQ, "coarse grid")
        if coarse_best["order"] is not None and coarse_best["sorder"] is not None:
            pdq_candidates = {coarse_best["order"]}
            spq_candidates = {coarse_best["sorder"]}
            for delta in NEIGHBOR_DELTAS:
                candidate_pdq = tuple(max(0, coarse_best["order"][i] + delta[i]) for i in range(3))
                candidate_spq = tuple(max(0, coarse_best["sorder"][i] + delta[i]) for i in range(3))
                if candidate_pdq in PDQ_SET:
                    pdq_candidates.add(candidate_pdq)
                if candidate_spq in SPQ_SET:
                    spq_candidates.add(candidate_spq)
            pdq_list = [combo for combo in PDQ if combo in pdq_candidates]
            spq_list = [combo for combo in SPQ if combo in spq_candidates]
            run_stage(pdq_list, spq_list, "refine grid")
    if best_global["order"] is None or best_global["metrics"]["RMSE"] == np.inf:
        run_stage(PDQ, SPQ, "full grid")

    best_global["tried"] = total_tried
    return best_global


# -----------------------------
# Train per meter
# -----------------------------
def train_for_meter(df_all, meter_col, meter_value, val_hours, outdir, save_model=False, grid_search_hours=0, progress_logger=None):
    if meter_col:
        df = df_all[df_all[meter_col] == meter_value].copy()
        if df.empty:
            print(f"[{meter_value}] no rows; skipping.")
            return
    else:
        df = df_all.copy()

    df = df.sort_values(TIMESTAMP).reset_index(drop=True)

    # Build exog list from columns present
    base_exog_cols = get_exog_cols(df)

    # Keep only needed cols to reduce memory issues
    keep_cols = [TIMESTAMP, TARGET] + base_exog_cols
    df = df[keep_cols].copy()

    # Split
    train, val = split_train_val(df, val_hours)

    tag = f"{meter_value}" if meter_col else "ALL_IN_ONE"
    grid_train = train
    if grid_search_hours and grid_search_hours > 0:
        subset_rows = min(grid_search_hours, len(train))
        if subset_rows < len(train):
            grid_train = train.iloc[-subset_rows:].copy()
            print(f"[{tag}] Using last {subset_rows}/{len(train)} rows for grid search.")
    grid_exog_cols, dropped_grid = split_constant_exog(grid_train, base_exog_cols)
    if dropped_grid:
        dropped_str = ", ".join(sorted(dropped_grid))
        print(f"[{tag}] Dropping constant exog for grid search: {dropped_str}")

    # Grid search
    best = grid_search(grid_train, val, grid_exog_cols, SEASONAL_PERIOD, meter_tag=tag, progress_logger=progress_logger)
    final_exog_cols, dropped_final = split_constant_exog(train, base_exog_cols)
    if dropped_final and set(dropped_final) != set(dropped_grid):
        dropped_str = ", ".join(sorted(dropped_final))
        print(f"[{tag}] Dropping constant exog for final fit: {dropped_str}")
    order = best["order"]
    sorder = best["sorder"]

    if order is None or sorder is None:
        tried = best.get('tried', 0) if isinstance(best, dict) else 0
        print(f"[{tag}] Grid search failed to find a valid model after trying {tried} combinations.")
        return

    # Refit on full train and forecast validation
    res, fcst, metrics, final_error = fit_and_forecast(train, val, final_exog_cols, order, sorder, SEASONAL_PERIOD)
    if res is None or fcst is None:
        message = (final_error or 'unknown error').replace('\n', ' ').replace('\r', ' ').strip()
        if len(message) > 200:
            message = message[:200] + '...'
        print(f"[{tag}] Final SARIMAX fit failed: {message}")
        return

    # Save outputs
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Predictions CSV
    fcst_values = np.asarray(fcst)
    fcst_len = len(fcst_values)
    pred_df = pd.DataFrame({
        "timestamp": val[TIMESTAMP].values[:fcst_len],
        "y_true": val[TARGET].values[:fcst_len],
        "y_pred": fcst_values
    })
    pred_path = outdir / f"sarimax_preds_{tag}.csv"
    pred_df.to_csv(pred_path, index=False)

    # Metrics JSON
    details = {
        "meter": tag,
        "order": order,
        "seasonal_order": (*sorder, SEASONAL_PERIOD),
        "val_hours": val_hours,
        "tried": best.get('tried', 0),
        "grid_search_rows": len(grid_train),
        "metrics": metrics,
        "exog_used": final_exog_cols,
        "grid_exog_used": grid_exog_cols,
        "dropped_exog_grid": dropped_grid,
        "dropped_exog_full": dropped_final
    }
    meta_path = outdir / f"sarimax_meta_{tag}.json"
    with open(meta_path, "w") as f:
        json.dump(details, f, indent=2)

    # Optional: save model
    if save_model and res is not None:
        try:
            model_path = outdir / f"sarimax_model_{tag}.pkl"
            res.save(model_path)
        except Exception as e:
            print(f"[{tag}] Could not save model: {e}")

    # Print a concise summary
    print(f"[{tag}] best order={order}, seasonal={(*sorder, SEASONAL_PERIOD)}, "
          f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, sMAPE={metrics['sMAPE']:.2f}%, "
          f"Tried {best.get('tried', 0)} combos.")
    print(f"Saved: {pred_path.name}, {meta_path.name}" + (f", sarimax_model_{tag}.pkl" if save_model else ""))


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="SARIMAX grid-search trainer (per meter).")
    parser.add_argument("--data", type=Path, default=Path(DATA_FILE), help="Path to dataset_timeseries.csv")
    parser.add_argument("--meter", type=str, default="M1",
                        help="Meter ID to train (e.g., M1). Use 'ALL' to loop all meters.")
    parser.add_argument("--val_hours", type=int, default=14*24, help="Validation window (hours). Default 14 days.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Base directory for outputs.")
    parser.add_argument("--grid_hours", type=int, default=0, help="If >0, use only the most recent N rows during grid-search.")
    parser.add_argument("--progress_log", type=str, default=None, help="Optional CSV file for grid-search progress.")
    parser.add_argument("--save_model", action="store_true", help="Save fitted model pickle.")
    parser.add_argument("--run_name", help="Optional name for this run; used for the output subdirectory.")
    parser.add_argument("--no_timestamp", action="store_true", help="Do not append an automatic timestamp to the run directory.")
    args = parser.parse_args()

    outdir = prepare_run_directory(
        args.outdir,
        args.run_name,
        timestamp=not args.no_timestamp,
    )
    print(f"Outputs will be written to {outdir}")

    progress_log_path = Path(args.progress_log) if args.progress_log else None
    if progress_log_path and not progress_log_path.is_absolute():
        progress_log_path = outdir / progress_log_path

    progress_logger = GridSearchProgressLogger(progress_log_path) if progress_log_path else None

    df, meter_col = load_data(args.data)
    if meter_col is None:
        print("No meter ID column detected; training on full series as one.")
        train_for_meter(df, None, None, args.val_hours, outdir,
                        save_model=args.save_model,
                        grid_search_hours=args.grid_hours,
                        progress_logger=progress_logger)
        return

    if args.meter.upper() == "ALL":
        meters = list(df[meter_col].dropna().unique())
        for m in meters:
            train_for_meter(df, meter_col, m, args.val_hours, outdir,
                            save_model=args.save_model,
                            grid_search_hours=args.grid_hours,
                            progress_logger=progress_logger)
    else:
        train_for_meter(df, meter_col, args.meter, args.val_hours, outdir,
                        save_model=args.save_model,
                        grid_search_hours=args.grid_hours,
                        progress_logger=progress_logger)


if __name__ == "__main__":
    main()









