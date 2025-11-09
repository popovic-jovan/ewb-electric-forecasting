"""Feature engineering helpers for aggregated time series models."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame, ts_col: str) -> Tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    timestamp = pd.to_datetime(df[ts_col])
    features = {
        "cal_hour": timestamp.dt.hour,
        "cal_dayofweek": timestamp.dt.dayofweek,
        "cal_is_weekend": timestamp.dt.dayofweek.isin([5, 6]).astype(int),
        "cal_month": timestamp.dt.month,
        "cal_dayofyear": timestamp.dt.dayofyear,
    }
    for name, values in features.items():
        df[name] = values
    return df, list(features.keys())


def add_cyclic_features(df: pd.DataFrame, ts_col: str) -> Tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    timestamp = pd.to_datetime(df[ts_col])
    hour = timestamp.dt.hour.to_numpy()
    dow = timestamp.dt.dayofweek.to_numpy()
    month = timestamp.dt.month.to_numpy()

    features = {
        "cyc_hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "cyc_hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "cyc_dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "cyc_dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "cyc_month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
        "cyc_month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
    }
    for name, values in features.items():
        df[name] = values
    return df, list(features.keys())


def add_fourier_terms(
    df: pd.DataFrame,
    ts_col: str,
    daily_k: int = 0,
    weekly_k: int = 0,
) -> Tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    timestamp = pd.to_datetime(df[ts_col])
    idx = timestamp.to_numpy(dtype="datetime64[ns]").astype(np.int64) // 10**9  # seconds since epoch
    features: list[str] = []

    def _generate(prefix: str, period: float, k_max: int):
        nonlocal df, features
        if k_max <= 0:
            return
        t = (idx - idx.min()) / 3600.0  # hours
        for k in range(1, k_max + 1):
            angle = 2 * np.pi * k * t / period
            sin_col = f"{prefix}_sin_{k}"
            cos_col = f"{prefix}_cos_{k}"
            df[sin_col] = np.sin(angle)
            df[cos_col] = np.cos(angle)
            features.extend([sin_col, cos_col])

    _generate("fourier24", 24.0, daily_k)
    _generate("fourier168", 168.0, weekly_k)
    return df, features


def add_target_lags(
    df: pd.DataFrame,
    target_col: str,
    lags: Sequence[int],
    group_col: str | None = None,
) -> Tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    feature_cols: list[str] = []
    grouped = df.groupby(group_col, group_keys=False) if group_col and group_col in df.columns else None
    for lag in sorted(set(lags or [])):
        col = f"{target_col}_lag_{lag}"
        if grouped is not None:
            df[col] = grouped[target_col].shift(lag)
        else:
            df[col] = df[target_col].shift(lag)
        feature_cols.append(col)
    return df, feature_cols


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    means: Sequence[int] | None = None,
    stds: Sequence[int] | None = None,
    diff_to_lag: int | None = None,
    group_col: str | None = None,
) -> Tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    feature_cols: list[str] = []
    grouped = df.groupby(group_col, group_keys=False) if group_col and group_col in df.columns else None
    if means:
        for window in means:
            col = f"{target_col}_roll_mean_{window}"
            if grouped is not None:
                df[col] = grouped[target_col].shift(1).transform(
                    lambda s: s.rolling(window, min_periods=max(1, window // 4)).mean()
                )
            else:
                df[col] = df[target_col].shift(1).rolling(window, min_periods=max(1, window // 4)).mean()
            feature_cols.append(col)
            if diff_to_lag:
                if grouped is not None:
                    ref = grouped[target_col].transform(lambda s: s.shift(diff_to_lag))
                else:
                    ref = df[target_col].shift(diff_to_lag)
                delta_col = f"{col}_delta_lag{diff_to_lag}"
                df[delta_col] = df[col] - ref
                feature_cols.append(delta_col)
    if stds:
        for window in stds:
            col = f"{target_col}_roll_std_{window}"
            if grouped is not None:
                df[col] = grouped[target_col].shift(1).transform(
                    lambda s: s.rolling(window, min_periods=max(2, window // 4)).std()
                )
            else:
                df[col] = df[target_col].shift(1).rolling(window, min_periods=max(2, window // 4)).std()
            feature_cols.append(col)
    return df, feature_cols


def build_time_series_features(
    df: pd.DataFrame,
    cfg: Mapping[str, object],
    ts_col: str,
    target_col: str,
    group_col: str | None = None,
) -> Tuple[pd.DataFrame, list[str]]:
    """Apply configured feature engineering steps to the aggregated dataframe."""
    features_cfg = cfg.get("features", {}) if cfg else {}
    exog_cols: list[str] = []
    if group_col and group_col in df.columns:
        df_feat = df.sort_values([group_col, ts_col]).copy()
    else:
        df_feat = df.sort_values(ts_col).copy()

    if features_cfg.get("calendar", True):
        df_feat, new = add_calendar_features(df_feat, ts_col)
        exog_cols.extend(new)
    if features_cfg.get("cyclic", True):
        df_feat, new = add_cyclic_features(df_feat, ts_col)
        exog_cols.extend(new)

    daily_k = int(features_cfg.get("fourier_daily_K", 0) or 0)
    weekly_k = int(features_cfg.get("fourier_weekly_K", 0) or 0)
    if daily_k > 0 or weekly_k > 0:
        df_feat, new = add_fourier_terms(df_feat, ts_col, daily_k=daily_k, weekly_k=weekly_k)
        exog_cols.extend(new)

    lags = features_cfg.get("target_lags", [])
    if lags:
        df_feat, new = add_target_lags(df_feat, target_col, lags, group_col=group_col)
        exog_cols.extend(new)

    roll_cfg = features_cfg.get("rolling", {})
    if isinstance(roll_cfg, Mapping):
        mean_windows = roll_cfg.get("mean") or roll_cfg.get("roll_mean")
        std_windows = roll_cfg.get("std") or roll_cfg.get("roll_std")
    else:
        mean_windows = features_cfg.get("roll_mean")
        std_windows = features_cfg.get("roll_std")

    if mean_windows or std_windows:
        diff_to_lag = None
        if isinstance(roll_cfg, Mapping):
            diff_to_lag = roll_cfg.get("diff_to_lag")
        elif features_cfg.get("roll_diff_to_lag"):
            diff_to_lag = features_cfg.get("roll_diff_to_lag")

        df_feat, new = add_rolling_features(
            df_feat,
            target_col=target_col,
            means=mean_windows or [],
            stds=std_windows or [],
            diff_to_lag=int(diff_to_lag) if diff_to_lag else None,
            group_col=group_col,
        )
        exog_cols.extend(new)

    # Holiday flags and derived signals (if present)
    if features_cfg.get("holiday_flags", True):
        holiday_cols = (
            "is_school_holiday",
            "is_public_holiday",
            "is_any_holiday",
            "school_holiday_run_hours",
            "public_holiday_run_hours",
            "any_holiday_run_hours",
        )
        for col in holiday_cols:
            if col in df_feat.columns:
                exog_cols.append(col)

    drop_columns = features_cfg.get("drop_columns") or []
    if drop_columns:
        drop_set = set(drop_columns)
        existing = [col for col in drop_set if col in df_feat.columns]
        if existing:
            df_feat = df_feat.drop(columns=existing)
        exog_cols = [col for col in exog_cols if col not in drop_set]

    return df_feat, exog_cols
