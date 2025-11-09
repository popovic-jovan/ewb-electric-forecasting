"""Diagnostics kit for investigating SARIMAX tuning failures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.core.data.preperation import load_aggregated_series, split_by_time_markers
from src.core.models.sarimax import SarimaxModel
from src.core.models.utils import assign_frequency, prepare_feature_splits, prepare_series_and_exog


def _read_yaml(path: str) -> dict:
    path_obj = Path(path)
    if not path_obj.exists():
        raise SystemExit(f"[DBG] Missing YAML file: {path}")
    with path_obj.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _coerce_exog(X: pd.DataFrame | None) -> pd.DataFrame | None:
    if X is None or X.empty:
        return None
    numeric = X.select_dtypes(include=[np.number]).astype("float64")
    numeric = numeric.loc[:, numeric.nunique(dropna=False) > 1]
    if numeric.shape[1] > 1:
        Q, R = np.linalg.qr(numeric.values)
        keep = np.where(np.abs(np.diag(R)) > 1e-10)[0]
        if len(keep) < numeric.shape[1]:
            numeric = numeric.iloc[:, keep]
    return numeric if not numeric.empty else None


def _fit(endog, exog, order, seasonal_order, trend, freq):
    model = SARIMAX(
        endog=endog,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
        dates=endog.index if isinstance(endog.index, (pd.DatetimeIndex, pd.PeriodIndex)) else None,
        freq=freq,
        initialization="approximate_diffuse",
        concentrate_scale=True,
    )
    try:
        return model.fit(method="lbfgs", disp=False, maxiter=200)
    except Exception:
        for method in ("powell", "nm", "bfgs"):
            try:
                return model.fit(method=method, disp=False, maxiter=400)
            except Exception:
                continue
        raise


def _dummy_family_report(columns: list[str]) -> dict[str, int]:
    families: dict[str, int] = {}
    expectations = {
        "hour": 24,
        "day": 7,
        "dayofweek": 7,
        "month": 12,
    }
    lower_cols = [col.lower() for col in columns]
    for prefix, expected in expectations.items():
        matches = [col for col in lower_cols if col.startswith(f"{prefix}_")]
        if matches:
            families[prefix] = len(matches)
            if len(matches) >= expected:
                families[prefix] = len(matches)
    return families


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose SARIMAX configuration issues.")
    parser.add_argument("--dataset-yaml", default="configs/dataset.yaml")
    parser.add_argument("--model-yaml", default="configs/model/sarimax.yaml")
    parser.add_argument("--ts-col", default="timestamp")
    parser.add_argument("--target-col", default="DELIVERED_VALUE")
    args = parser.parse_args()

    dataset_cfg = _read_yaml(args.dataset_yaml)
    model_cfg = _read_yaml(args.model_yaml)

    aggregated = load_aggregated_series(dataset_cfg)
    frames = split_by_time_markers(aggregated, dataset_cfg, ts_col=args.ts_col)

    feature_cfg = model_cfg.get("features", {})
    train_df, val_df, _, exog_cols = prepare_feature_splits(frames, feature_cfg, args.ts_col, args.target_col)
    train_df = SarimaxModel._ensure_datetime_index(train_df, args.ts_col)
    val_df = SarimaxModel._ensure_datetime_index(val_df, args.ts_col)

    use_log = bool(model_cfg.get("use_log1p", False))
    freq_hint = assign_frequency([train_df, val_df], dataset_cfg.get("freq"))

    y_train, X_train = prepare_series_and_exog(train_df, args.target_col, exog_cols, use_log)
    y_val, X_val = prepare_series_and_exog(val_df, args.target_col, exog_cols, use_log)

    if y_train.empty:
        raise SystemExit("[DBG] Training series is empty; cannot run probes.")

    if X_train is None or X_train.empty:
        print("[DBG] EXOG: none (pure SARIMA).")
    else:
        non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        zero_var = X_train.columns[X_train.nunique(dropna=False) <= 1].tolist()
        rank = np.linalg.matrix_rank(X_train.select_dtypes(include=[np.number]).values)
        print(
            f"[DBG] EXOG cols={X_train.shape[1]}, non-numeric={len(non_numeric)}, "
            f"zero-variance={len(zero_var)}, rank={rank}/{X_train.shape[1]}"
        )
        if non_numeric:
            print("[DBG] Non-numeric columns:", non_numeric[:15])
        if zero_var:
            print("[DBG] Zero-variance columns:", zero_var[:15])
        families = _dummy_family_report(X_train.columns.tolist())
        if families:
            print("[DBG] Dummy family sizes:", families)

    try:
        inferred = pd.infer_freq(y_train.index)
    except Exception:
        inferred = None
    steps = (
        y_train.index.to_series().diff().value_counts().head(5).to_dict()
        if len(y_train) > 1
        else {}
    )
    span = (
        (y_train.index[-1] - y_train.index[0]).total_seconds() / 3600.0
        if len(y_train) > 1
        else 0.0
    )
    print(f"[DBG] infer_freq={inferred}, len(y_train)={len(y_train)}, span_hours={span:.2f}")
    print("[DBG] Top index steps:", steps)

    seasonal_period = int(model_cfg.get("seasonal_period", 24) or 24)
    if len(y_train) < 5 * max(1, seasonal_period):
        print(
            f"[WARN] Train length {len(y_train)} may be too short for seasonal period m={seasonal_period}."
        )

    print(f"[PROBE A] Fit (0,0,0)x(1,0,0,{seasonal_period}) WITHOUT EXOG, trend='n'")
    _fit(y_train, None, (0, 0, 0), (1, 0, 0, seasonal_period), "n", freq_hint)
    print("[PROBE A] SUCCESS")

    sanitized = _coerce_exog(X_train)
    if sanitized is not None:
        sanitized = sanitized.reindex(y_train.index)
        if sanitized.isna().any().any():
            sanitized = sanitized.ffill().bfill()
        rank = np.linalg.matrix_rank(sanitized.values) if sanitized.shape[1] else 0
        print(f"[DBG] Sanitized EXOG shape={sanitized.shape}, rank={rank}")
        print(f"[PROBE B] Fit (0,0,0)x(1,0,0,{seasonal_period}) WITH sanitized EXOG, trend='n'")
        _fit(y_train, sanitized, (0, 0, 0), (1, 0, 0, seasonal_period), "n", freq_hint)
        print("[PROBE B] SUCCESS")
    else:
        print("[PROBE B] Skipped (no usable EXOG).")


if __name__ == "__main__":
    main()

