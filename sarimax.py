# train_sarimax.py
# -------------------------------------------
# SARIMAX with grid search on hourly electricity data (per meter).
# Exogenous regressors: weather + calendar/cyclical features from dataset_timeseries.csv
#
# Usage:
#   python train_sarimax.py --meter M1 --val_hours 336 --save_model
#   python train_sarimax.py --meter ALL --val_hours 336
# -------------------------------------------
import argparse
import json
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


# -----------------------------
# Config
# -----------------------------
DATA_FILE = "model_datasets/dataset_timeseries.csv"
TARGET = "delivered_value"
TIMESTAMP = "timestamp"
ID_COLS_CANDIDATES = ["meter_ui", "nmi_ui"]

# Default grid (sensible + not insane)
PDQ = [(p, d, q) for p in [0, 1, 2] for d in [0, 1] for q in [0, 1, 2]]
SPQ = [(P, D, Q) for P in [0, 1] for D in [0, 1] for Q in [0, 1, 2]]
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
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100
    s = smape(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": s}


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
    exog_train = train[exog_cols] if exog_cols else None
    exog_val = val[exog_cols] if exog_cols else None

    # Drop any rows with missing in target/exog
    if exog_cols:
        train_ = train[[TARGET] + exog_cols].dropna().copy()
        val_ = val[[TARGET] + exog_cols].dropna().copy()
        # Align exog to the filtered indices
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

    if len(y_train) < (seasonal_period * 2):
        # not enough data to fit seasonal model robustly
        return None, None, {"MAE": np.inf, "RMSE": np.inf, "MAPE": np.inf, "sMAPE": np.inf}

    try:
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=order,
            seasonal_order=(sorder[0], sorder[1], sorder[2], seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend="c"
        )
        res = model.fit(maxiter=maxiter, disp=False)
        # Forecast horizon = len(val_)
        fcst = res.forecast(steps=len(val_), exog=X_val)
        metrics = evaluate(y_val.values, fcst.values)
        return res, fcst, metrics
    except Exception:
        # non-convergence or numerical issues
        return None, None, {"MAE": np.inf, "RMSE": np.inf, "MAPE": np.inf, "sMAPE": np.inf}


def grid_search(train, val, exog_cols, seasonal_period):
    best = {"order": None, "sorder": None, "metrics": {"RMSE": np.inf}}
    tried = 0
    for order in PDQ:
        for sorder in SPQ:
            tried += 1
            res, fcst, metrics = fit_and_forecast(train, val, exog_cols, order, sorder, seasonal_period)
            if metrics["RMSE"] < best["metrics"]["RMSE"]:
                best = {"order": order, "sorder": sorder, "metrics": metrics}
    best["tried"] = tried
    return best


# -----------------------------
# Train per meter
# -----------------------------
def train_for_meter(df_all, meter_col, meter_value, val_hours, outdir, save_model=False):
    if meter_col:
        df = df_all[df_all[meter_col] == meter_value].copy()
        if df.empty:
            print(f"[{meter_value}] no rows; skipping.")
            return
    else:
        df = df_all.copy()

    df = df.sort_values(TIMESTAMP).reset_index(drop=True)

    # Build exog list from columns present
    exog_cols = get_exog_cols(df)

    # Keep only needed cols to reduce memory issues
    keep_cols = [TIMESTAMP, TARGET] + exog_cols
    df = df[keep_cols].copy()

    # Split
    train, val = split_train_val(df, val_hours)

    # Grid search
    best = grid_search(train, val, exog_cols, SEASONAL_PERIOD)
    order = best["order"]
    sorder = best["sorder"]

    # Refit on full train and forecast validation
    res, fcst, metrics = fit_and_forecast(train, val, exog_cols, order, sorder, SEASONAL_PERIOD)

    # Save outputs
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{meter_value}" if meter_col else "ALL_IN_ONE"

    # Predictions CSV
    pred_df = pd.DataFrame({
        "timestamp": val[TIMESTAMP].values[:len(fcst)],
        "y_true": val[TARGET].values[:len(fcst)],
        "y_pred": fcst.values
    })
    pred_path = outdir / f"sarimax_preds_{tag}.csv"
    pred_df.to_csv(pred_path, index=False)

    # Metrics JSON
    details = {
        "meter": tag,
        "order": order,
        "seasonal_order": (*sorder, SEASONAL_PERIOD),
        "val_hours": val_hours,
        "tried": best["tried"],
        "metrics": metrics,
        "exog_used": exog_cols
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
          f"Tried {best['tried']} combos.")
    print(f"Saved: {pred_path.name}, {meta_path.name}" + (f", sarimax_model_{tag}.pkl" if save_model else ""))


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="SARIMAX grid-search trainer (per meter).")
    parser.add_argument("--data", type=str, default=DATA_FILE, help="Path to dataset_timeseries.csv")
    parser.add_argument("--meter", type=str, default="M1",
                        help="Meter ID to train (e.g., M1). Use 'ALL' to loop all meters.")
    parser.add_argument("--val_hours", type=int, default=14*24, help="Validation window (hours). Default 14 days.")
    parser.add_argument("--outdir", type=str, default="sarimax_out", help="Output directory.")
    parser.add_argument("--save_model", action="store_true", help="Save fitted model pickle.")
    args = parser.parse_args()

    df, meter_col = load_data(args.data)
    if meter_col is None:
        print("No meter ID column detected; training on full series as one.")
        train_for_meter(df, None, None, args.val_hours, args.outdir, save_model=args.save_model)
        return

    if args.meter.upper() == "ALL":
        meters = list(df[meter_col].dropna().unique())
        for m in meters:
            train_for_meter(df, meter_col, m, args.val_hours, args.outdir, save_model=args.save_model)
    else:
        train_for_meter(df, meter_col, args.meter, args.val_hours, args.outdir, save_model=args.save_model)


if __name__ == "__main__":
    main()
