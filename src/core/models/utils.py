"""Shared helpers for aggregated time-series model implementations."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from core.data.preperation import SplitFrames
from src.core.features.time_series import build_time_series_features


def prepare_feature_splits(
    frames: SplitFrames,
    feature_cfg: Mapping[str, object],
    ts_col: str,
    target_col: str,
    group_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    def _tag(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            reference_cols: Sequence[str] = []
            for candidate in (frames.train, frames.val, frames.test):
                if candidate is not None and not candidate.empty:
                    reference_cols = candidate.columns
                    break
            empty = pd.DataFrame(columns=list(reference_cols))
            empty["__split"] = pd.Series(dtype="object")
            return empty.iloc[0:0]
        tagged = frame.copy()
        tagged["__split"] = label
        return tagged

    combined = pd.concat(
        [_tag(frames.train, "train"), _tag(frames.val, "val"), _tag(frames.test, "test")],
        ignore_index=True,
    )

    if combined.empty:
        return frames.train.copy(), frames.val.copy(), frames.test.copy(), []

    combined, exog_cols = build_time_series_features(
        combined,
        {"features": feature_cfg},
        ts_col=ts_col,
        target_col=target_col,
        group_col=group_col,
    )

    combined = combined.dropna(subset=[target_col])
    if exog_cols:
        combined = combined.dropna(subset=exog_cols)

    splits: dict[str, pd.DataFrame] = {}
    for label in ("train", "val", "test"):
        subset = combined[combined["__split"] == label].drop(columns="__split", errors="ignore").copy()
        subset = subset.reset_index(drop=True)
        splits[label] = subset

    return splits["train"], splits["val"], splits["test"], exog_cols


def prepare_series_and_exog(
    df: pd.DataFrame,
    target_col: str,
    exog_cols: Sequence[str],
    use_log: bool = False,
) -> Tuple[pd.Series, pd.DataFrame | None]:
    if df.empty:
        return pd.Series(dtype=float), None
    y = df[target_col]
    if use_log:
        y = np.log1p(np.clip(y, a_min=0, a_max=None))
    X = df[list(exog_cols)] if exog_cols else None
    return y, X


def inverse_log(values: pd.Series | np.ndarray, use_log: bool) -> np.ndarray:
    if not use_log:
        return np.asarray(values)
    return np.expm1(np.asarray(values))


def assign_frequency(
    frames: Sequence[pd.DataFrame],
    freq_pref: str | None = None,
) -> str | None:
    """Attach frequency metadata to datetime indexes and return the resolved freq string."""
    freq_hint = freq_pref
    offset = None

    if isinstance(freq_hint, str):
        try:
            offset = to_offset(freq_hint)
            freq_hint = offset.freqstr
        except ValueError:
            offset = None

    if not freq_hint:
        for frame in frames:
            if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
                continue
            try:
                inferred = pd.infer_freq(frame.index)
            except (TypeError, ValueError):
                inferred = None
            if inferred:
                freq_hint = inferred
                try:
                    offset = to_offset(inferred)
                except ValueError:
                    offset = None
                break

    if offset is not None:
        for frame in frames:
            if frame is None or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
                continue
            frame.index = pd.DatetimeIndex(frame.index, freq=offset)

    return freq_hint
