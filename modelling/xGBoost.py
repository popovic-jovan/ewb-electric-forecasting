import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from sklearn.model_selection import ParameterGrid

import joblib
import numpy as np
import pandas as pd
import warnings
from pandas.api.types import CategoricalDtype
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "xgboost is required. Install with `pip install xgboost`."
    ) from exc

try:
    from .output_utils import prepare_run_directory
except ImportError:  # pragma: no cover - fallback when run as script
    from output_utils import prepare_run_directory

DATA_FILE = "datasets/dataset_xgboost.csv"
TARGET = "y"
TIMESTAMP = "timestamp"
METER_COLUMN = "meter_ui"
DEFAULT_OUTPUT_DIR = Path("models/xgboost")

DEFAULT_DROP_COLS = {
    "timestamp",
    "meter_ui",
   # "maximum_temperature_degree_c",
    #"maximum_temperature_degree_c_tmax_mean_168h_prevday",
   # "minimum_temperature_degree_c",
  #   "temp_mean",
#    "temp_range",
#    "CDH",
#    "solar_lag_24",
     "rain_lag_24",
 "rain_lag_72",
# "rain_lag_168",
 "rainfall_amount_millimetres_rain_cum_72h_prevday",
#"rainfall_amount_millimetres_rain_cum_168h_prevday",
}



def smape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1e-9
    return 100 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)


def evaluate(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    mase_scale: Optional[float] = None,
) -> Dict[str, float]:

    mae = mean_absolute_error(y_true, y_pred)

    try:

        rmse = mean_squared_error(y_true, y_pred, squared=False)

    except TypeError:  # compatibility with older sklearn

        rmse = mean_squared_error(y_true, y_pred) ** 0.5

    y_true_arr = np.asarray(y_true)

    y_pred_arr = np.asarray(y_pred)

    abs_true = np.abs(y_true_arr)

    abs_err = np.abs(y_true_arr - y_pred_arr)

    denom = np.maximum(abs_true, 1e-9)

    mape = np.mean(abs_err / denom) * 100

    total_actual = abs_true.sum()

    if total_actual <= 1e-9:

        wape = float("nan")

    else:

        wape = abs_err.sum() / total_actual * 100

    if mase_scale is not None and mase_scale > 0:

        mase = mae / mase_scale

    else:

        mase = float("nan")

    return {

        "MAE": mae,

        "RMSE": rmse,

        "MAPE": mape,

        "sMAPE": smape(y_true_arr, y_pred_arr),

        "WAPE": wape,

        "MASE": mase,

        "R2": r2_score(y_true, y_pred),

    }






def compute_mase_scale(
    df: pd.DataFrame,
    target_col: str,
    meter_col: Optional[str],
    seasonality: int = 1,
) -> Optional[float]:
    if seasonality <= 0:
        raise ValueError("mase_seasonality must be positive.")
    diffs: List[np.ndarray] = []
    if meter_col and meter_col in df.columns:
        for _, group in df.groupby(meter_col):
            values = group[target_col].to_numpy(dtype=float)
            if len(values) <= seasonality:
                continue
            diff = np.abs(values[seasonality:] - values[:-seasonality])
            if diff.size:
                diffs.append(diff)
    else:
        values = df[target_col].to_numpy(dtype=float)
        if len(values) > seasonality:
            diff = np.abs(values[seasonality:] - values[:-seasonality])
            if diff.size:
                diffs.append(diff)
    if not diffs:
        return None
    return float(np.mean(np.concatenate(diffs)))



def load_dataset(
    path: Path,
    timestamp_col: str,
    meter_col: str,
    meter: Optional[str],
    dropna_target: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Dataset must include timestamp column '{timestamp_col}'.")
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must include target column '{TARGET}'.")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    if dropna_target:
        df = df.dropna(subset=[TARGET])

    if meter is not None:
        if meter_col not in df.columns:
            raise ValueError(f"Meter column '{meter_col}' not present in dataset.")
        df = df[df[meter_col] == meter]
        if df.empty:
            raise ValueError(f"No rows found for {meter_col}='{meter}'.")

    df = df.sort_values([meter_col, timestamp_col]) if meter_col in df.columns else df.sort_values(timestamp_col)
    return df.reset_index(drop=True)


def build_features(
    df: pd.DataFrame,
    target_col: str,
    drop_columns: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    drop_set = set(drop_columns)
    available_drop = [col for col in drop_set if col in df.columns]
    always_drop = {"timestamp"}
    extra_drop = [col for col in always_drop if col in df.columns]
    X = df.drop(columns=available_drop + extra_drop + [target_col])

    datetime_cols = X.select_dtypes(include=["datetime", "datetimetz"]).columns
    if len(datetime_cols):
        X = X.drop(columns=list(datetime_cols))

    y = df[target_col]
    return X, y


def train_val_split(
    df: pd.DataFrame,
    meter_col: Optional[str],
    timestamp_col: str,
    val_hours: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if val_hours <= 0:
        raise ValueError("val_hours must be positive.")
    if meter_col and meter_col in df.columns:
        train_parts = []
        val_parts = []
        for _, group in df.groupby(meter_col, sort=False):
            if len(group) <= val_hours:
                raise ValueError(
                    f"Meter '{group.iloc[0][meter_col]}' has only {len(group)} rows; cannot hold out {val_hours} validation hours."
                )
            train_parts.append(group.iloc[:-val_hours])
            val_parts.append(group.iloc[-val_hours:])
        train_df = pd.concat(train_parts, ignore_index=True)
        val_df = pd.concat(val_parts, ignore_index=True)
    else:
        if len(df) <= val_hours:
            raise ValueError("Dataset too small for requested val_hours.")
        train_df = df.iloc[:-val_hours]
        val_df = df.iloc[-val_hours:]
    return train_df, val_df


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, object],
) -> xgb.XGBRegressor:
    train_object_cols = X_train.select_dtypes(include=["object", "string"]).columns
    for col in train_object_cols:
        train_cat = X_train[col].astype("category")
        if col in X_val.columns:
            val_cat = X_val[col].astype("category")
            categories = train_cat.cat.categories.union(val_cat.cat.categories)
            X_train[col] = train_cat.cat.set_categories(categories)
            X_val[col] = val_cat.cat.set_categories(categories)
        else:
            X_train[col] = train_cat

    extra_val_object_cols = [
        col for col in X_val.select_dtypes(include=["object", "string"]).columns if col not in train_object_cols
    ]
    for col in extra_val_object_cols:
        X_val[col] = X_val[col].astype("category")

    categorical_cols = [
        col for col in X_train.columns if isinstance(X_train[col].dtype, CategoricalDtype)
    ]

    def _build_model(cfg: Dict[str, object]) -> xgb.XGBRegressor:
        local = cfg.copy()
        tree_method = local.pop("tree_method", "hist")
        device = local.pop("device", None)
        if tree_method == "gpu_hist":
            tree_method = "hist"
            if device != "cpu":
                device = "cuda"

        model_kwargs = dict(
            objective="reg:squarederror",
            tree_method=tree_method,
            n_estimators=int(local.pop("n_estimators", 500)),
            learning_rate=float(local.pop("eta", 0.05)),
            max_depth=int(local.pop("max_depth", 6)),
            subsample=float(local.pop("subsample", 0.8)),
            colsample_bytree=float(local.pop("colsample_bytree", 0.8)),
            reg_lambda=float(local.pop("reg_lambda", 1.0)),
            reg_alpha=float(local.pop("reg_alpha", 0.0)),
            min_child_weight=float(local.pop("min_child_weight", 1.0)),
            random_state=int(local.pop("random_state", 42)),
            **local,
        )
        if device:
            model_kwargs["device"] = device
        if categorical_cols:
            model_kwargs["enable_categorical"] = True
        return xgb.XGBRegressor(**model_kwargs)

    try:
        model = _build_model(params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        return model
    except xgb.core.XGBoostError as exc:
        if params.get("device") == "cuda":
            warnings.warn(
                f"GPU training failed with error '{exc}'. Falling back to CPU (device='cpu').",
                RuntimeWarning,
            )
            params["tree_method"] = "hist"
            params["device"] = "cpu"
            model = _build_model(params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False,
            )
            return model
        raise


def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    base_params: Dict[str, object],
    grid_options: Dict[str, List[float]],
    mase_scale: Optional[float],
) -> Tuple[xgb.XGBRegressor, Dict[str, object], Dict[str, float], List[Dict[str, object]]]:
    search_space = {k: v for k, v in grid_options.items() if v}
    if not search_space:
        raise ValueError("Grid search requested but no grid_* values were provided.")

    best_score = float('inf')
    best_model: Optional[xgb.XGBRegressor] = None
    best_params = base_params.copy()
    best_metrics: Optional[Dict[str, float]] = None
    results: List[Dict[str, object]] = []

    for combo in ParameterGrid(search_space):
        trial_params = base_params.copy()
        trial_params.update({k: combo[k] for k in combo})
        if trial_params.get("tree_method") == "gpu_hist":
            trial_params["tree_method"] = "hist"
            if trial_params.get("device", base_params.get("device")) != "cpu":
                trial_params["device"] = "cuda"
        model = train_model(X_train, y_train, X_val, y_val, trial_params)
        preds = model.predict(X_val)
        metrics = evaluate(y_val, preds, mase_scale=mase_scale)
        results.append({"params": trial_params.copy(), "metrics": metrics})
        score = metrics["RMSE"]
        if score < best_score:
            best_score = score
            best_model = model
            best_params = trial_params.copy()
            best_metrics = metrics

    if best_model is None or best_metrics is None:
        raise RuntimeError("Grid search failed to evaluate any parameter combinations.")

    return best_model, best_params, best_metrics, results



def save_outputs(
    output_dir: Path,
    model: xgb.XGBRegressor,
    predictions: pd.DataFrame,
    metrics: Dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBoost regressor on the ML dataset.")
    parser.add_argument("--data_path", type=Path, default=Path(DATA_FILE), help="Path to the input CSV data.")
    parser.add_argument("--timestamp", default=TIMESTAMP, help="Timestamp column name.")
    parser.add_argument("--target", default=TARGET, help="Target column to predict.")
    parser.add_argument("--meter_column", default=METER_COLUMN, help="Meter identifier column name.")
    parser.add_argument("--meter", help="Specific meter identifier to filter before training.")
    parser.add_argument("--val_hours", type=int, default=168, help="Number of trailing hours per meter for validation.")
    parser.add_argument(
        "--drop_columns",
        nargs="*",
        default=sorted(DEFAULT_DROP_COLS),
        help="Columns to drop before training (in addition to the target).",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Base directory to store outputs.")
    parser.add_argument("--run_name", help="Optional name for this run; becomes the subdirectory under the output base.")
    parser.add_argument("--no_timestamp", action="store_true", help="Do not append an automatic timestamp to the run directory.")
    parser.add_argument("--n_estimators", type=int, default=500, help="Number of boosting rounds.")
    parser.add_argument("--eta", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--max_depth", type=int, default=6, help="Maximum tree depth.")
    parser.add_argument("--subsample", type=float, default=0.8, help="Row subsampling ratio.")
    parser.add_argument("--colsample_bytree", type=float, default=0.8, help="Column subsampling per tree.")
    parser.add_argument("--reg_lambda", type=float, default=1.0, help="L2 regularisation term on weights.")
    parser.add_argument("--reg_alpha", type=float, default=0.0, help="L1 regularisation term on weights.")
    parser.add_argument("--min_child_weight", type=float, default=1.0, help="Minimum sum of instance weight (hessian) needed in a child.")
    parser.add_argument(
        "--tree_method",
        default="hist",
        help="Tree construction algorithm (default 'hist').",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="XGBoost device (e.g., 'cuda' or 'cpu'). Default uses CUDA with automatic CPU fallback.",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument("--dropna_target", action="store_true", help="Drop rows where the target is NA before splitting.")
    parser.add_argument("--mase_seasonality", type=int, default=1, help="Seasonality period for MASE scaling (1 = naive benchmark).")
    parser.add_argument("--grid_search", action="store_true", help="Enable grid search over provided grid_* options using the validation split.")
    parser.add_argument("--grid_n_estimators", type=int, nargs="*", help="n_estimators values to try when grid search is enabled.")
    parser.add_argument("--grid_eta", type=float, nargs="*", help="Learning rate values to try during grid search.")
    parser.add_argument("--grid_max_depth", type=int, nargs="*", help="max_depth values to try during grid search.")
    parser.add_argument("--grid_subsample", type=float, nargs="*", help="subsample ratios to try during grid search.")
    parser.add_argument("--grid_colsample_bytree", type=float, nargs="*", help="colsample_bytree values to try during grid search.")
    parser.add_argument("--grid_reg_lambda", type=float, nargs="*", help="reg_lambda values to try during grid search.")
    parser.add_argument("--grid_reg_alpha", type=float, nargs="*", help="reg_alpha values to try during grid search.")
    parser.add_argument("--grid_min_child_weight", type=float, nargs="*", help="min_child_weight values to try during grid search.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    args.tree_method = (args.tree_method or "hist").lower()
    args.device = (args.device or "cuda").lower()
    if args.tree_method == "gpu_hist":
        warnings.warn(
            "XGBoost 2.0 deprecates tree_method='gpu_hist'; using tree_method='hist' with device='cuda'.",
            RuntimeWarning,
        )
        args.tree_method = "hist"
        if args.device != "cpu":
            args.device = "cuda"
    if args.device == "auto":
        args.device = "cuda"

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
        meter_col=args.meter_column,
        meter=args.meter,
        dropna_target=args.dropna_target,
    )

    train_df, val_df = train_val_split(
        df=df,
        meter_col=args.meter_column if args.meter is None else None,
        timestamp_col=args.timestamp,
        val_hours=args.val_hours,
    )

    X_train, y_train = build_features(train_df, args.target, args.drop_columns)
    X_val, y_val = build_features(val_df, args.target, args.drop_columns)

    def _convert_categoricals(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        object_cols = train_df.select_dtypes(include=["object", "string"]).columns
        for col in object_cols:
            train_cat = train_df[col].astype("category")
            if col in val_df.columns:
                val_cat = val_df[col].astype("category")
                categories = train_cat.cat.categories.union(val_cat.cat.categories)
                train_df[col] = train_cat.cat.set_categories(categories)
                val_df[col] = val_cat.cat.set_categories(categories)
            else:
                train_df[col] = train_cat

        extra_val_cols = [
            col
            for col in val_df.select_dtypes(include=["object", "string"]).columns
            if col not in object_cols
        ]
        for col in extra_val_cols:
            val_df[col] = val_df[col].astype("category")

    _convert_categoricals(X_train, X_val)

    meter_for_scale = args.meter_column if (args.meter is None and args.meter_column in train_df.columns) else None
    mase_scale = compute_mase_scale(
        train_df,
        args.target,
        meter_for_scale,
        seasonality=args.mase_seasonality,
    )

    params = {
        "n_estimators": args.n_estimators,
        "eta": args.eta,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "min_child_weight": args.min_child_weight,
        "tree_method": args.tree_method,
        "device": args.device,
        "random_state": args.random_state,
    }

    grid_results = None
    best_params_used = params.copy()

    if args.grid_search:
        grid_options = {
            "n_estimators": args.grid_n_estimators,
            "eta": args.grid_eta,
            "max_depth": args.grid_max_depth,
            "subsample": args.grid_subsample,
            "colsample_bytree": args.grid_colsample_bytree,
            "reg_lambda": args.grid_reg_lambda,
            "reg_alpha": args.grid_reg_alpha,
            "min_child_weight": args.grid_min_child_weight,
        }
        model, best_params_used, best_metrics, grid_results = run_grid_search(
            X_train,
            y_train,
            X_val,
            y_val,
            params,
            grid_options,
            mase_scale,
        )
        val_pred = model.predict(X_val)
        metrics = best_metrics
    else:
        model = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            params,
        )
        val_pred = model.predict(X_val)
        metrics = evaluate(y_val, val_pred, mase_scale=mase_scale)
        best_params_used = params.copy()

    predictions = val_df[[args.timestamp]].copy()
    predictions["y_true"] = y_val.values
    predictions["y_pred"] = val_pred
    if args.meter_column in val_df.columns:
        predictions[args.meter_column] = val_df[args.meter_column].values

    output_dir = args.output_dir
    if args.meter:
        output_dir = output_dir / f"{args.meter_column}_{args.meter}"

    save_outputs(output_dir, model, predictions, metrics)

    best_params_path = output_dir / "best_params.json"
    with best_params_path.open("w", encoding="utf-8") as f:
        json.dump(best_params_used, f, indent=2)

    if grid_results is not None:
        with (output_dir / "grid_results.json").open("w", encoding="utf-8") as f:
            json.dump(grid_results, f, indent=2)
        print(f"Grid search evaluated {len(grid_results)} combination(s).")
        print(f"Best parameters: {json.dumps(best_params_used, indent=2)}")
    else:
        print(f"Parameters: {json.dumps(best_params_used, indent=2)}")
    print(f"Validation metrics: {json.dumps(metrics, indent=2)}")
    print(f"Saved model to {output_dir / 'model.joblib'}")
    print(f"Saved predictions to {output_dir / 'predictions.csv'}")
    print(f"Saved parameters to {best_params_path}")


if __name__ == "__main__":
    main()






