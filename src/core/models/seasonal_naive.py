"""Seasonal naïve baseline model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.core.data.preperation import load_aggregated_series, load_raw_series, split_by_time_markers
from src.core.evaluation.metrics import metric_dict
from src.core.models import ModelBase, ModelInfo, TrainResult
from src.core.registry import register


def _prepare_frame(
    df: pd.DataFrame,
    target_col: str,
    freq: str | None,
    group_col: str | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[target_col])

    frame = df.copy()
    if "timestamp" not in frame.columns:
        raise KeyError("Expected 'timestamp' column in aggregated dataframe.")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    timestamps = frame["timestamp"]

    if getattr(timestamps.dt, "tz", None) is not None:
        frame["timestamp"] = timestamps.dt.tz_convert(None)

    if group_col and group_col in frame.columns:
        frame[group_col] = frame[group_col].astype(str)
        frame = frame.sort_values([group_col, "timestamp"])
        frame = frame.set_index([group_col, "timestamp"])
    else:
        frame = frame.set_index("timestamp").sort_index()
        if freq:
            try:
                frame = frame.asfreq(freq.lower())
            except Exception:
                pass

    return frame


def _seasonal_naive(
    series: pd.Series,
    seasonal_period: int,
    group_level: str | None = None,
) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    if group_level:
        return series.groupby(level=group_level).shift(seasonal_period)
    return series.shift(seasonal_period)


def _collect_metrics(
    frames: Mapping[str, pd.DataFrame],
    forecasts: Mapping[str, pd.Series],
    target_col: str,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for split, df in frames.items():
        if df is None or df.empty:
            continue
        actual = df[target_col]
        preds = forecasts.get(split, pd.Series(dtype=float)).reindex(actual.index)
        mask = preds.notna() & actual.notna()
        if not mask.any():
            continue
        results[split] = metric_dict(actual[mask], preds[mask])
    return results


@register
class SeasonalNaiveModel(ModelBase):
    info = ModelInfo(
        name="seasonal_naive",
        display_name="Seasonal Naïve Baseline",
        default_train_config=Path("configs/model/seasonal_naive.yaml"),
        default_tune_config=None,
        description="Baseline that repeats the value from one seasonal period ago.",
        tags=("baseline", "time-series"),
    )

    def train(self, data: pd.DataFrame, output_dir: Path) -> TrainResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = output_dir / "reports"
        predictions_dir = output_dir / "predictions"
        reports_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)

        target_col = self.dataset_config.get("target_col", "DELIVERED_VALUE")
        freq = self.dataset_config.get("freq", "H")
        freq = freq.lower() if isinstance(freq, str) else None
        seasonal_period = int(self.config.get("seasonal_period", 24))
        dataset_mode = str(self.config.get("dataset_mode", "aggregated")).lower()
        group_col: str | None = None

        if dataset_mode == "raw":
            source_df = load_raw_series(self.dataset_config, data)
            group_col = self.dataset_config.get("id_col")
            if not group_col or group_col not in source_df.columns:
                raise ValueError("Raw dataset mode requires a valid 'id_col' column in the data.")
        else:
            source_df = load_aggregated_series(self.dataset_config, data)

        frames_split = split_by_time_markers(source_df, self.dataset_config)

        train_df = _prepare_frame(frames_split.train, target_col, freq, group_col=group_col)
        val_df = _prepare_frame(frames_split.val, target_col, freq, group_col=group_col)
        test_df = _prepare_frame(frames_split.test, target_col, freq, group_col=group_col)
        frames = {"Train": train_df, "Val": val_df, "Test": test_df}

        forecasts = {
            split: _seasonal_naive(df[target_col], seasonal_period, group_level=group_col)
            for split, df in frames.items()
        }

        metrics = _collect_metrics(frames, forecasts, target_col)
        metrics_path: Path | None = None
        if metrics:
            records = [
                {"model": "seasonal_naive", "split": split, **values}
                for split, values in metrics.items()
            ]
            metrics_df = pd.DataFrame.from_records(records)
            metrics_path = reports_dir / "metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)

        prediction_frames = []
        for split, df in frames.items():
            if df.empty:
                continue
            preds = forecasts[split].reindex(df.index)
            df_reset = df.reset_index()
            df_reset["split"] = split
            df_reset["y"] = df[target_col].to_numpy()
            df_reset["y_hat"] = preds.to_numpy()

            ordered_cols = ["split"]
            if group_col and group_col in df_reset.columns:
                ordered_cols.append(group_col)
            if "timestamp" in df_reset.columns:
                ordered_cols.append("timestamp")
            else:
                ordered_cols.append("index")
            ordered_cols.extend(["y", "y_hat"])
            prediction_frames.append(df_reset[ordered_cols])
        predictions_path: Path | None = None
        if prediction_frames:
            predictions_path = predictions_dir / "seasonal_naive_predictions.csv"
            pd.concat(prediction_frames).to_csv(predictions_path, index=False)

        artifacts: dict[str, Path] = {"predictions_dir": predictions_dir}
        if metrics_path:
            artifacts["metrics"] = metrics_path
        if predictions_path:
            artifacts["predictions"] = predictions_path

        # use validation/test metrics for summary
        summary_metrics = metrics.get("Test", metrics.get("Val", {}))
        return TrainResult(
            fitted_model=None,
            metrics=summary_metrics,
            artifacts=artifacts,
            model_path=None,
        )

    def tune(self, data: pd.DataFrame, output_dir: Path) -> dict[str, float]:
        raise NotImplementedError("Seasonal naive baseline does not support tuning.")

    def predict(
        self,
        model_path: Path,
        data: pd.DataFrame,
        horizon: int | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError("Use 'train' output predictions for seasonal naive baseline.")
