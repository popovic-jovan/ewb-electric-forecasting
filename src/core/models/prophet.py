"""Aggregated Prophet model implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

from core.data.preperation import load_aggregated_series, load_raw_series, split_by_time_markers
from src.core.models import ModelBase, ModelInfo, TrainResult
from src.core.registry import register
from src.core.evaluation.metrics import metric_dict


def _to_prophet_frame(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, object]:
    if df.empty:
        empty = pd.DataFrame({"ds": pd.Series(dtype="datetime64[ns]"), "y": pd.Series(dtype=float)})
        return empty, None

    ts = pd.to_datetime(df["timestamp"])
    tz = getattr(ts.dt, "tz", None)
    if tz is not None:
        ts = ts.dt.tz_convert(None)
    frame = pd.DataFrame({"ds": ts, "y": df[target_col].astype(float).values})
    return frame, tz


def _restore_timezone(series: pd.Series, tz) -> pd.Series:
    if tz is None:
        return series
    return series.dt.tz_localize(tz)


@register
class ProphetModel(ModelBase):
    info = ModelInfo(
        name="prophet",
        display_name="Prophet",
        default_train_config=Path("configs/model/prophet.yaml"),
        default_tune_config=Path("configs/model/prophet.yaml"),
        description="Facebook Prophet applied to aggregated hourly series.",
        tags=("time-series", "prophet"),
    )

    def __init__(self, config: Mapping[str, object], dataset_config: Mapping[str, object]):
        super().__init__(config, dataset_config)
        self._active_meter: str | None = None

    def _prepare_frames(self, data: pd.DataFrame):
        target_col = self.dataset_config.get("target_col", "DELIVERED_VALUE")
        dataset_mode = str(self.config.get("dataset_mode", "aggregated")).lower()
        if dataset_mode == "raw":
            aggregated, meter_id = self._load_single_meter_frame(data)
            self._active_meter = meter_id
        else:
            aggregated = load_aggregated_series(self.dataset_config, data)
            self._active_meter = None
        frames = split_by_time_markers(aggregated, self.dataset_config)
        return target_col, frames

    def train(self, data: pd.DataFrame, output_dir: Path) -> TrainResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = output_dir / "artifacts"
        reports_dir = output_dir / "reports"
        predictions_dir = output_dir / "predictions"
        for path in (artifacts_dir, reports_dir, predictions_dir):
            path.mkdir(parents=True, exist_ok=True)

        target_col, frames = self._prepare_frames(data)

        df_train, tz_train = _to_prophet_frame(frames.train, target_col)
        df_val, tz_val = _to_prophet_frame(frames.val, target_col)
        df_test, tz_test = _to_prophet_frame(frames.test, target_col)

        quick_mode = bool(self.config.get("_runtime_quick"))
        prophet_params = dict(self.config.get("prophet_params", {}))
        if quick_mode:
            prophet_params.setdefault("seasonality_mode", "additive")
            prophet_params.setdefault("weekly_seasonality", False)
            prophet_params.setdefault("daily_seasonality", False)
            prophet_params.setdefault("n_changepoints", 10)

        model = Prophet(**prophet_params)
        for reg in self.config.get("extra_regressors", []):
            model.add_regressor(reg)

        if df_train.empty:
            raise ValueError("Training data is empty after aggregation; cannot fit Prophet model.")

        model.fit(df_train)

        def _predict(frame: pd.DataFrame, tz) -> pd.Series:
            if frame.empty:
                return pd.Series(dtype=float)
            preds = model.predict(frame[["ds"]])
            yhat = preds.set_index("ds")["yhat"]
            if tz is not None:
                yhat.index = yhat.index.tz_localize(tz)
            return yhat

        preds_train = _predict(df_train, tz_train)
        preds_val = _predict(df_val, tz_val)
        preds_test = _predict(df_test, tz_test)

        def _reindex(frame: pd.DataFrame, preds: pd.Series) -> pd.Series:
            if frame.empty:
                return pd.Series(dtype=float)
            ts = pd.to_datetime(frame["ds"])
            if getattr(ts.dt, "tz", None) is None and tz_train is not None:
                ts = ts.dt.tz_localize(tz_train)
            return preds.reindex(ts)

        train_series = df_train.set_index("ds")["y"]
        val_series = df_val.set_index("ds")["y"] if not df_val.empty else pd.Series(dtype=float)
        test_series = df_test.set_index("ds")["y"] if not df_test.empty else pd.Series(dtype=float)

        if tz_train is not None:
            train_series.index = train_series.index.tz_localize(tz_train)
        if tz_val is not None and not val_series.empty:
            val_series.index = val_series.index.tz_localize(tz_val)
        if tz_test is not None and not test_series.empty:
            test_series.index = test_series.index.tz_localize(tz_test)

        metrics: dict[str, dict[str, float]] = {}
        metrics["Train"] = metric_dict(train_series, preds_train.reindex(train_series.index))
        if not val_series.empty:
            metrics["Val"] = metric_dict(val_series, preds_val.reindex(val_series.index))
        if not test_series.empty:
            metrics["Test"] = metric_dict(test_series, preds_test.reindex(test_series.index))

        metrics_path = reports_dir / "metrics.csv"
        metric_records = [{"split": split, **values} for split, values in metrics.items()]
        if self._active_meter:
            for record in metric_records:
                record["meter"] = self._active_meter
        pd.DataFrame.from_records(metric_records).to_csv(metrics_path, index=False)

        def _save_preds(series: pd.Series, split: str) -> None:
            if series.empty:
                return
            df_out = series.to_frame("y_hat")
            if self._active_meter:
                df_out["meter"] = self._active_meter
            df_out.to_csv(predictions_dir / f"{split}.csv")

        _save_preds(preds_train, "train")
        _save_preds(preds_val, "val")
        _save_preds(preds_test, "test")

        model_json = model_to_json(model)
        model_path = artifacts_dir / "prophet_model.json"
        model_path.write_text(model_json, encoding="utf-8")

        primary_metrics = metrics.get("Test", metrics.get("Val", {}))
        return TrainResult(
            fitted_model=None,
            metrics=primary_metrics,
            artifacts={
                "metrics": metrics_path,
                "predictions_dir": predictions_dir,
            },
            model_path=model_path,
        )

    def tune(self, data: pd.DataFrame, output_dir: Path) -> dict[str, float]:
        output_dir.mkdir(parents=True, exist_ok=True)
        tuning_dir = output_dir / "tuning"
        tuning_dir.mkdir(parents=True, exist_ok=True)

        target_col, frames = self._prepare_frames(data)
        df_train, tz_train = _to_prophet_frame(frames.train, target_col)
        df_val, tz_val = _to_prophet_frame(frames.val, target_col)

        if df_val.empty:
            raise RuntimeError("Validation data is required for Prophet tuning; adjust dataset splits.")

        quick_mode = bool(self.config.get("_runtime_quick"))
        base_params = dict(self.config.get("prophet_params", {}))
        grid = self.config.get("tune_grid", {})
        if quick_mode:
            grid = {key: values[:1] for key, values in grid.items()} if grid else {}

        if not grid:
            grid = {
                "changepoint_prior_scale": [base_params.get("changepoint_prior_scale", 0.05)],
                "seasonality_prior_scale": [base_params.get("seasonality_prior_scale", 10.0)],
            }

        trials: list[dict[str, float]] = []
        best_score = float("inf")
        best_params = base_params

        for cp in grid.get("changepoint_prior_scale", [base_params.get("changepoint_prior_scale", 0.05)]):
            for sp in grid.get("seasonality_prior_scale", [base_params.get("seasonality_prior_scale", 10.0)]):
                params = dict(base_params)
                params["changepoint_prior_scale"] = cp
                params["seasonality_prior_scale"] = sp
                model = Prophet(**params)
                model.fit(df_train)
                preds_val = model.predict(df_val[["ds"]])
                yhat = preds_val.set_index("ds")["yhat"]
                val_series = df_val.set_index("ds")["y"]
                if tz_val is not None:
                    yhat.index = yhat.index.tz_localize(tz_val)
                    val_series.index = val_series.index.tz_localize(tz_val)
                score = metric_dict(val_series, yhat.reindex(val_series.index))["WAPE"]
                trials.append(
                    {
                        "changepoint_prior_scale": cp,
                        "seasonality_prior_scale": sp,
                        "WAPE": score,
                    }
                )
                if score < best_score:
                    best_score = score
                    best_params = params

        trials_path = tuning_dir / "prophet_trials.csv"
        pd.DataFrame(trials).to_csv(trials_path, index=False)

        best_path = tuning_dir / "prophet_best.json"
        with best_path.open("w", encoding="utf-8") as handle:
            json.dump({"params": best_params, "WAPE": best_score}, handle, indent=2)

        return {"WAPE": best_score}

    def predict(
        self,
        model_path: Path,
        data: pd.DataFrame,
        horizon: int | None = None,
    ) -> pd.DataFrame:
        model_json = model_path.read_text(encoding="utf-8")
        model = model_from_json(model_json)

        target_col, frames = self._prepare_frames(data)
        aggregated = pd.concat([frames.train, frames.val, frames.test], ignore_index=True)
        aggregated = aggregated.sort_values("timestamp")

        ds = pd.to_datetime(aggregated["timestamp"])
        tz = getattr(ds.dt, "tz", None)
        if tz is not None:
            ds = ds.dt.tz_convert(None)

        future = pd.DataFrame({"ds": ds})
        preds = model.predict(future)
        yhat = preds.set_index("ds")["yhat"]
        if tz is not None:
            yhat.index = yhat.index.tz_localize(tz)

        df_out = pd.DataFrame(
            {
                "timestamp": yhat.index,
                "y_hat": yhat.values,
            }
        )
        if target_col in aggregated.columns:
            actual = aggregated[target_col].values.astype(float)
            df_out["y"] = actual
        if self._active_meter:
            df_out["meter"] = self._active_meter
        return df_out

    def _load_single_meter_frame(self, data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        id_col = self.dataset_config.get("id_col")
        if not id_col:
            raise ValueError("Raw dataset mode requires 'id_col' in dataset configuration.")

        df = load_raw_series(self.dataset_config, data)
        if id_col not in df.columns:
            raise ValueError(f"Column '{id_col}' missing from raw dataset; cannot isolate meter.")

        df[id_col] = df[id_col].astype(str)
        selected = self.config.get("_selected_meters") or self.dataset_config.get("_selected_meters")
        if selected:
            allowed = {str(item) for item in selected}
            df = df[df[id_col].isin(allowed)]

        unique_ids = df[id_col].unique()
        if len(unique_ids) != 1:
            raise ValueError(
                "Prophet raw mode expects data for exactly one meter. "
                "Provide --meter to select a single identifier."
            )
        meter_id = unique_ids[0]
        meter_df = df[df[id_col] == meter_id].copy()
        return meter_df, meter_id
