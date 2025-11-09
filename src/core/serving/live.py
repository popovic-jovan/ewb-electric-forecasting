"""Helpers for serving live forecasts in the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Sequence

import json
import pandas as pd
from pandas.tseries.frequencies import to_offset

from core.data.preperation import load_aggregated_series, load_raw_series
from src.core.features.time_series import build_time_series_features
from src.core.io import load_yaml
from src.core.models import XGBModel


@dataclass
class LiveSnapshot:
    """Container representing the latest observed usage and a one-step forecast."""

    dataset_mode: str
    meter_id: str | None
    current_timestamp: pd.Timestamp
    current_value: float
    previous_value: float | None
    forecast_timestamp: pd.Timestamp
    forecast_value: float
    flags: Mapping[str, bool]
    cost_per_kwh: float
    current_cost: float
    forecast_cost: float
    today_cost: float
    usage_threshold: float
    current_over_threshold: bool
    forecast_over_threshold: bool
    history: pd.DataFrame
    future_features: pd.Series


def _ensure_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        return ts.dt.tz_localize(None)
    return ts


def _compute_special_day_flags(ts: pd.Timestamp, dataset_cfg: Mapping[str, object]) -> dict[str, int]:
    special_cfg = dataset_cfg.get("special_days") or {}
    if ts.tzinfo is not None:
        ts_local = ts.tz_convert(None)
    else:
        ts_local = ts
    day = ts_local.normalize()

    public_holidays = special_cfg.get("public_holidays") or []
    public_days = {
        pd.Timestamp(item).normalize()
        for item in public_holidays
        if not pd.isna(pd.to_datetime(item, errors="coerce"))
    }
    is_public = int(day in public_days)

    is_school = 0
    for entry in special_cfg.get("school_holidays") or []:
        start = pd.to_datetime(entry.get("start"), errors="coerce")
        end = pd.to_datetime(entry.get("end"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        start = start.normalize()
        end = end.normalize()
        if start <= day <= end:
            is_school = 1
            break

    is_any = int(bool(is_public or is_school))

    return {
        "is_public_holiday": is_public,
        "is_school_holiday": is_school,
        "is_any_holiday": is_any,
    }


def _next_run(prev_flag: int, prev_run: float, current_flag: int) -> float:
    if current_flag:
        return float((prev_run if prev_flag else 0) + 1)
    return 0.0


def _append_future_rows(
    df: pd.DataFrame,
    dataset_cfg: Mapping[str, object],
    dataset_mode: str,
    group_col: str | None,
    meter_id: str | None,
    steps: int,
) -> tuple[pd.DataFrame, pd.tseries.offsets.BaseOffset]:
    ts_col = "timestamp"
    target_col = dataset_cfg.get("target_col", "DELIVERED_VALUE")
    freq = dataset_cfg.get("freq", "H")
    offset = to_offset(freq.lower() if isinstance(freq, str) else freq)

    df_ext = df.copy()
    df_ext[ts_col] = _ensure_timestamp(df_ext[ts_col])
    df_ext = df_ext.sort_values(ts_col)
    for col in (
        "is_public_holiday",
        "is_school_holiday",
        "is_any_holiday",
        "public_holiday_run_hours",
        "school_holiday_run_hours",
        "any_holiday_run_hours",
    ):
        if col not in df_ext.columns:
            df_ext[col] = 0

    for step in range(1, steps + 1):
        future_ts = df_ext[ts_col].max() + offset * step
        flags = _compute_special_day_flags(future_ts, dataset_cfg)
        future_row: MutableMapping[str, object] = {
            ts_col: future_ts,
            target_col: float("nan"),
            "is_public_holiday": flags["is_public_holiday"],
            "is_school_holiday": flags["is_school_holiday"],
            "is_any_holiday": flags["is_any_holiday"],
        }

        if group_col:
            future_row[group_col] = meter_id
            mask = df_ext[group_col].astype(str) == str(meter_id)
            prev_rows = df_ext[mask]
        else:
            prev_rows = df_ext

        if prev_rows.empty:
            prev_public_flag = prev_school_flag = prev_any_flag = 0
            prev_public_run = prev_school_run = prev_any_run = 0.0
        else:
            prev = prev_rows.iloc[-1]
            prev_public_flag = int(prev.get("is_public_holiday", 0) or 0)
            prev_school_flag = int(prev.get("is_school_holiday", 0) or 0)
            prev_any_flag = int(prev.get("is_any_holiday", prev_public_flag or prev_school_flag) or 0)
            prev_public_run = float(prev.get("public_holiday_run_hours", 0) or 0)
            prev_school_run = float(prev.get("school_holiday_run_hours", 0) or 0)
            prev_any_run = float(prev.get("any_holiday_run_hours", 0) or 0)

        future_row["public_holiday_run_hours"] = _next_run(
            prev_public_flag, prev_public_run, flags["is_public_holiday"]
        )
        future_row["school_holiday_run_hours"] = _next_run(
            prev_school_flag, prev_school_run, flags["is_school_holiday"]
        )
        future_row["any_holiday_run_hours"] = _next_run(
            prev_any_flag, prev_any_run, flags["is_any_holiday"]
        )

        df_ext = pd.concat([df_ext, pd.DataFrame([future_row])], ignore_index=True, sort=False)

    return df_ext, offset


def list_meter_ids(dataset_cfg_path: Path) -> list[str]:
    """Return the set of available meter identifiers from the raw dataset."""
    dataset_cfg = load_yaml(dataset_cfg_path)
    id_col = dataset_cfg.get("id_col")
    if not id_col:
        return []
    csv_path = dataset_cfg.get("raw_csv")
    if not csv_path:
        return []
    df = pd.read_csv(csv_path, usecols=[id_col])
    return sorted(df[id_col].astype(str).unique())


def load_xgb_model(model_path: Path) -> tuple[XGBModel, Sequence[str]]:
    """Load a trained XGBoost model and its feature column order."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    feature_path = model_path.with_name("feature_columns.json")
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature column metadata missing at {feature_path}; re-run training to regenerate it."
        )

    model = XGBModel.load(model_path)
    feature_cols = json.loads(feature_path.read_text(encoding="utf-8"))
    return model, feature_cols


def load_xgb_live_snapshot(
    model: XGBModel,
    feature_columns: Sequence[str],
    dataset_cfg: Mapping[str, object],
    model_cfg: Mapping[str, object],
    meter_id: str | None = None,
    history_hours: int = 24,
    focus_timestamp: pd.Timestamp | None = None,
) -> LiveSnapshot:
    """Prepare the latest observation and a one-step-ahead XGBoost forecast."""
    dataset_mode = str(model_cfg.get("dataset_mode", "aggregated")).lower()
    ts_col = "timestamp"
    target_col = dataset_cfg.get("target_col", "DELIVERED_VALUE")
    id_col = dataset_cfg.get("id_col")

    raw_df = pd.read_csv(dataset_cfg["raw_csv"])
    if dataset_mode == "raw":
        if not id_col:
            raise ValueError("Raw dataset mode requires 'id_col' in dataset configuration.")
        if meter_id is None:
            raise ValueError("Meter identifier must be specified when dataset_mode='raw'.")
        raw_df[id_col] = raw_df[id_col].astype(str)
        raw_df = raw_df[raw_df[id_col] == str(meter_id)]
        if raw_df.empty:
            raise ValueError(f"No rows found for meter '{meter_id}'.")
        base_series = load_raw_series(dataset_cfg, raw_df)
        group_col = id_col
    else:
        base_series = load_aggregated_series(dataset_cfg, raw_df)
        group_col = None

    base_series[ts_col] = _ensure_timestamp(base_series[ts_col])
    base_series = base_series.sort_values(ts_col)

    if base_series.empty:
        raise ValueError("Dataset is empty after preprocessing; unable to produce snapshot.")

    ts_series = base_series[ts_col]
    tz = getattr(ts_series.dt, "tz", None)
    if focus_timestamp is not None:
        focus_ts = pd.to_datetime(focus_timestamp)
        if tz is not None:
            if focus_ts.tzinfo is None:
                focus_ts = focus_ts.tz_localize(tz)
            else:
                focus_ts = focus_ts.tz_convert(tz)
    else:
        focus_ts = ts_series.max()

    if focus_ts > ts_series.max():
        raise ValueError("Focus timestamp is beyond available data.")

    subset_series = base_series[base_series[ts_col] <= focus_ts].copy()
    if subset_series.empty or subset_series[ts_col].max() < focus_ts:
        raise ValueError("Focus timestamp not present in dataset.")

    extended_series, offset = _append_future_rows(
        subset_series,
        dataset_cfg=dataset_cfg,
        dataset_mode=dataset_mode,
        group_col=group_col,
        meter_id=meter_id,
        steps=1,
    )

    features_cfg = {"features": model_cfg.get("features", {})}
    feature_frame, _ = build_time_series_features(
        extended_series,
        features_cfg,
        ts_col=ts_col,
        target_col=target_col,
        group_col=group_col,
    )

    feature_frame[ts_col] = _ensure_timestamp(feature_frame[ts_col])
    feature_frame = feature_frame.sort_values([group_col, ts_col] if group_col else ts_col)

    if group_col:
        feature_frame[group_col] = feature_frame[group_col].astype(str)
        feature_frame = feature_frame[feature_frame[group_col] == str(meter_id)]

    future_ts = focus_ts + offset
    future_mask = feature_frame[ts_col] == future_ts
    if future_mask.sum() == 0:
        raise RuntimeError("Unable to compute feature vector for the forecast horizon (insufficient history?).")

    future_row = feature_frame.loc[future_mask].tail(1)
    feature_values = future_row[feature_columns].iloc[0]
    if feature_values.isna().any():
        missing = [col for col in feature_columns if pd.isna(feature_values[col])]
        raise RuntimeError(f"Feature vector for future timestamp contains NaNs: {missing}")

    forecast_value = float(model.predict(future_row[feature_columns])[0])

    actual_rows = feature_frame[(feature_frame[target_col].notna()) & (feature_frame[ts_col] <= focus_ts)]
    if actual_rows.empty:
        raise RuntimeError("No valid historical rows after feature engineering.")
    current_row = actual_rows.tail(1)
    current_ts = current_row[ts_col].iloc[0]
    current_value = float(current_row[target_col].iloc[0])
    previous_row = actual_rows.tail(2).head(1)
    previous_value = float(previous_row[target_col].iloc[0]) if not previous_row.empty else None

    history_window = actual_rows.tail(history_hours).copy()
    history_window = history_window.dropna(subset=feature_columns, how="any")
    history_predictions = model.predict(history_window[feature_columns])

    history_df = pd.DataFrame(
        {
            "timestamp": history_window[ts_col].values,
            "Actual": history_window[target_col].values,
            "Forecast": history_predictions,
        }
    )
    forecast_append = pd.DataFrame(
        {
            "timestamp": [future_ts],
            "Actual": [None],
            "Forecast": [forecast_value],
        }
    )
    forecast_append = forecast_append.astype({"Actual": float, "Forecast": float})
    history_df = pd.concat([history_df, forecast_append], ignore_index=True)

    flag_cols = {
        "is_public_holiday": bool(current_row.get("is_public_holiday", pd.Series([0])).iloc[0]),
        "is_school_holiday": bool(current_row.get("is_school_holiday", pd.Series([0])).iloc[0]),
        "is_any_holiday": bool(current_row.get("is_any_holiday", pd.Series([0])).iloc[0]),
    }

    pricing_cfg = model_cfg.get("pricing", {})
    cost_per_kwh = float(pricing_cfg.get("cost_per_kwh", 0.315))
    daily_supply_charge = float(pricing_cfg.get("daily_supply_charge", 1.322))
    usage_threshold = float(pricing_cfg.get("usage_warning_threshold_kwh", 2.2))
    current_over_threshold = current_value > usage_threshold
    forecast_over_threshold = forecast_value > usage_threshold
    current_cost = current_value * cost_per_kwh
    forecast_cost = forecast_value * cost_per_kwh

    actual_ts = _ensure_timestamp(actual_rows[ts_col])
    current_day = _ensure_timestamp(pd.Series([current_ts])).iloc[0].normalize()
    today_mask = actual_ts.dt.normalize() == current_day
    today_consumption = float(actual_rows.loc[today_mask, target_col].sum())
    today_cost = today_consumption * cost_per_kwh + daily_supply_charge

    return LiveSnapshot(
        dataset_mode=dataset_mode,
        meter_id=str(meter_id) if meter_id is not None else None,
        current_timestamp=current_ts,
        current_value=current_value,
        previous_value=previous_value,
        forecast_timestamp=future_ts,
        forecast_value=forecast_value,
        flags=flag_cols,
        cost_per_kwh=cost_per_kwh,
        current_cost=current_cost,
        forecast_cost=forecast_cost,
        today_cost=today_cost,
        usage_threshold=usage_threshold,
        current_over_threshold=current_over_threshold,
        forecast_over_threshold=forecast_over_threshold,
        history=history_df,
        future_features=future_values_with_index(feature_values),
    )


def future_values_with_index(feature_series: pd.Series) -> pd.Series:
    """Ensure feature vector carries a helpful name for display."""
    feature_series = feature_series.copy().astype(float)
    feature_series.name = "next_hour_features"
    return feature_series
