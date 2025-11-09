"""Aggregated SARIMA implementation without exogenous regressors."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from core.data.preperation import load_aggregated_series, load_raw_series, split_by_time_markers
from src.core.evaluation.metrics import metric_dict, wape
from src.core.models import ModelBase, ModelInfo, TrainResult
from src.core.models.sarimax import deserialize_model, fit_sarimax, safe_predict, serialize_model
from src.core.models.utils import assign_frequency, inverse_log
from src.core.registry import register


def _grid_orders(cfg: Mapping[str, object]) -> Iterable[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    search = cfg.get("search", {}) if cfg else {}
    p_opts = search.get("p", [0])
    d_opts = search.get("d", [0])
    q_opts = search.get("q", [0])
    P_opts = search.get("P", [0])
    D_opts = search.get("D", [0])
    Q_opts = search.get("Q", [0])
    seasonal_period = int(cfg.get("seasonal_period", 24))
    seasonal_enabled = bool(search.get("seasonal", True))

    seasonal_orders = [(0, 0, 0, 0)]
    if seasonal_enabled and seasonal_period > 0:
        seasonal_orders = [
            (int(P), int(D), int(Q), seasonal_period)
            for P, D, Q in itertools.product(P_opts, D_opts, Q_opts)
        ]

    orders = [
        (int(p), int(d), int(q))
        for p, d, q in itertools.product(p_opts, d_opts, q_opts)
    ]

    if cfg.get("_runtime_quick"):
        orders = orders[: min(3, len(orders))]
        seasonal_orders = seasonal_orders[: min(2, len(seasonal_orders))]

    for order in orders:
        for seasonal_order in seasonal_orders:
            yield order, seasonal_order


@dataclass
class ForecastArtifacts:
    train: pd.Series
    val: pd.Series
    test: pd.Series


@register
class SarimaModel(ModelBase):
    info = ModelInfo(
        name="sarima",
        display_name="SARIMA (aggregated)",
        default_train_config=Path("configs/model/sarima.yaml"),
        default_tune_config=Path("configs/model/sarima.yaml"),
        description="Seasonal ARIMA applied to the aggregated load series.",
        tags=("time-series", "statsmodels"),
    )

    def __init__(self, config: Mapping[str, object], dataset_config: Mapping[str, object]):
        super().__init__(config, dataset_config)
        self._active_meter: str | None = None

    def _prepare_frames(self, data: pd.DataFrame):
        target_col = self.dataset_config.get("target_col", "DELIVERED_VALUE")
        dataset_mode = str(self.config.get("dataset_mode", "aggregated")).lower()
        if dataset_mode == "raw":
            aggregated, meter_id = self._load_single_meter_series(data)
            self._active_meter = meter_id
        else:
            aggregated = load_aggregated_series(self.dataset_config, data)
            self._active_meter = None
        frames = split_by_time_markers(aggregated, self.dataset_config)
        assign_frequency(
            [frames.train.set_index("timestamp"), frames.val.set_index("timestamp"), frames.test.set_index("timestamp")],
            self.dataset_config.get("freq"),
        )
        return target_col, frames

    def _load_single_meter_series(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        id_col = self.dataset_config.get("id_col")
        if not id_col:
            raise ValueError("Raw dataset mode requires 'id_col' in the dataset configuration.")

        source_df = load_raw_series(self.dataset_config, data)
        if id_col not in source_df.columns:
            raise ValueError(f"Column '{id_col}' not found in raw dataset; cannot isolate meter.")

        source_df[id_col] = source_df[id_col].astype(str)
        selected = (
            self.config.get("_selected_meters")
            or self.dataset_config.get("_selected_meters")
        )
        if selected:
            allowed = {str(item) for item in selected}
            source_df = source_df[source_df[id_col].isin(allowed)]

        unique_ids = source_df[id_col].unique()
        if len(unique_ids) != 1:
            raise ValueError(
                "SARIMA raw mode expects data for exactly one meter. "
                "Provide a meter identifier via --meter."
            )
        meter_id = unique_ids[0]
        meter_df = source_df[source_df[id_col] == meter_id].copy()
        return meter_df, meter_id

    @staticmethod
    def _set_freq(series: pd.Series, freq_hint: str | None) -> pd.Series:
        if series.empty or not freq_hint:
            return series
        try:
            offset = to_offset(freq_hint)
            series.index = pd.DatetimeIndex(series.index, freq=offset)
        except (ValueError, TypeError):
            pass
        return series

    def train(self, data: pd.DataFrame, output_dir: Path) -> TrainResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = output_dir / "artifacts"
        reports_dir = output_dir / "reports"
        predictions_dir = output_dir / "predictions"
        for path in (artifacts_dir, reports_dir, predictions_dir):
            path.mkdir(parents=True, exist_ok=True)

        target_col, frames = self._prepare_frames(data)

        use_log = bool(self.config.get("use_log1p", False))

        def _series(frame: pd.DataFrame) -> pd.Series:
            if frame.empty:
                return pd.Series(dtype=float)
            s = frame.set_index("timestamp")[target_col].astype(float)
            return s

        train_series = _series(frames.train)
        val_series = _series(frames.val)
        test_series = _series(frames.test)

        series_list = [train_series, val_series, test_series]
        freq_hint = assign_frequency([s.to_frame() for s in series_list if not s.empty], self.dataset_config.get("freq"))
        train_series = self._set_freq(train_series, freq_hint)
        val_series = self._set_freq(val_series, freq_hint)
        test_series = self._set_freq(test_series, freq_hint)

        def _prepare(series: pd.Series) -> pd.Series:
            if series.empty:
                return series
            if use_log:
                return np.log1p(np.clip(series, a_min=0, a_max=None))
            return series

        y_train = _prepare(train_series)
        y_val = _prepare(val_series)
        y_test = _prepare(test_series)

        best_score = math.inf
        best_combo = None
        best_result = None

        for order, seasonal_order in _grid_orders(self.config):
            print(f"[SARIMA] Evaluating order={order}, seasonal={seasonal_order}")
            try:
                result = fit_sarimax(
                    y_train,
                    exog=None,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=self.config.get("trend", "c"),
                    freq=freq_hint,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[SARIMA] Failed to fit order={order} seasonal={seasonal_order}: {exc}")
                continue

            if val_series.empty:
                best_combo = (order, seasonal_order)
                best_result = result
                break

            try:
                preds = safe_predict(result, start=val_series.index[0], end=val_series.index[-1])
                preds = pd.Series(preds, index=val_series.index)
                preds = pd.Series(inverse_log(preds, use_log), index=preds.index)
                score = wape(val_series.values, preds.values)
            except Exception as exc:  # noqa: BLE001
                print(f"[SARIMA] Validation prediction failed order={order} seasonal={seasonal_order}: {exc}")
                continue

            if np.isnan(score):
                continue
            if score < best_score:
                best_score = score
                best_combo = (order, seasonal_order)
                best_result = result

        if best_result is None or best_combo is None:
            raise RuntimeError("SARIMA grid search failed to converge on any configuration.")

        combined_series = pd.concat([train_series, val_series]).sort_index()
        y_combined = _prepare(combined_series)
        best_result = fit_sarimax(
            y_combined,
            exog=None,
            order=best_combo[0],
            seasonal_order=best_combo[1],
            trend=self.config.get("trend", "c"),
            freq=freq_hint,
        )

        forecasts = self._generate_forecasts(best_result, train_series, val_series, test_series, use_log)
        metrics = self._collect_metrics(train_series, val_series, test_series, forecasts)

        metrics_path = reports_dir / "metrics.csv"
        metric_records = []
        for split, values in metrics.items():
            record = {"split": split, **values}
            if self._active_meter:
                record["meter"] = self._active_meter
            metric_records.append(record)
        pd.DataFrame.from_records(metric_records).to_csv(metrics_path, index=False)

        def _save_forecast(series: pd.Series, split: str) -> None:
            if series.empty:
                return
            df_out = series.to_frame("y_hat")
            if self._active_meter:
                df_out["meter"] = self._active_meter
            df_out.to_csv(predictions_dir / f"{split}.csv")

        _save_forecast(forecasts.train, "train")
        _save_forecast(forecasts.val, "val")
        _save_forecast(forecasts.test, "test")

        model_path = artifacts_dir / "sarima_model.pkl"
        model_path.write_bytes(serialize_model(best_result))

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
        train_series = frames.train.set_index("timestamp")[target_col].astype(float)
        val_series = frames.val.set_index("timestamp")[target_col].astype(float)

        if val_series.empty:
            raise RuntimeError("Validation data is required for SARIMA tuning; adjust dataset splits.")

        freq_hint = assign_frequency([train_series.to_frame(), val_series.to_frame()], self.dataset_config.get("freq"))
        use_log = bool(self.config.get("use_log1p", False))

        y_train = np.log1p(np.clip(train_series, a_min=0, a_max=None)) if use_log else train_series
        y_val = np.log1p(np.clip(val_series, a_min=0, a_max=None)) if use_log else val_series

        trials = []
        best_score = float("inf")
        best_combo = None

        for order, seasonal_order in _grid_orders(self.config):
            try:
                result = fit_sarimax(
                    y_train,
                    exog=None,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=self.config.get("trend", "c"),
                    freq=freq_hint,
                )
                preds = safe_predict(result, start=val_series.index[0], end=val_series.index[-1])
                preds = pd.Series(preds, index=val_series.index)
                preds = pd.Series(inverse_log(preds, use_log), index=preds.index)
                score = wape(val_series.values, preds.values)
            except Exception as exc:  # noqa: BLE001
                print(f"[SARIMA] Tuning candidate failed ({order}, {seasonal_order}): {exc}")
                continue

            record = {
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "P": seasonal_order[0],
                "D": seasonal_order[1],
                "Q": seasonal_order[2],
                "m": seasonal_order[3],
                "WAPE": score,
            }
            trials.append(record)
            if np.isnan(score):
                continue
            if score < best_score:
                best_score = score
                best_combo = (order, seasonal_order)

        trials_path = tuning_dir / "sarima_trials.csv"
        pd.DataFrame(trials).to_csv(trials_path, index=False)

        if best_combo is None:
            raise RuntimeError("Tuning failed to produce a valid SARIMA configuration.")

        best_path = tuning_dir / "sarima_best.json"
        pd.DataFrame(
            [
                {
                    "p": best_combo[0][0],
                    "d": best_combo[0][1],
                    "q": best_combo[0][2],
                    "P": best_combo[1][0],
                    "D": best_combo[1][1],
                    "Q": best_combo[1][2],
                    "m": best_combo[1][3],
                    "WAPE": best_score,
                }
            ]
        ).to_json(best_path, orient="records", indent=2)
        return {"WAPE": best_score}

    def predict(
        self,
        model_path: Path,
        data: pd.DataFrame,
        horizon: int | None = None,
    ) -> pd.DataFrame:
        result = deserialize_model(model_path.read_bytes())
        target_col, frames = self._prepare_frames(data)
        aggregated = pd.concat([frames.train, frames.val, frames.test], ignore_index=True)
        aggregated = aggregated.sort_values("timestamp")

        series = aggregated.set_index("timestamp")[target_col].astype(float)
        assign_frequency([series.to_frame()], self.dataset_config.get("freq"))

        preds = safe_predict(result, start=series.index[0], end=series.index[-1])
        preds = pd.Series(preds, index=series.index)
        if self.config.get("use_log1p", False):
            preds = pd.Series(inverse_log(preds, True), index=preds.index)

        df_out = pd.DataFrame(
            {
                "timestamp": preds.index,
                "y_hat": preds.values,
                "y": series.values,
            }
        )
        return df_out

    @staticmethod
    def _generate_forecasts(
        result,
        train_series: pd.Series,
        val_series: pd.Series,
        test_series: pd.Series,
        use_log: bool,
    ) -> ForecastArtifacts:
        def _predict(series: pd.Series) -> pd.Series:
            if series.empty:
                return pd.Series(dtype=float)
            preds = safe_predict(result, start=series.index[0], end=series.index[-1])
            preds = pd.Series(preds, index=series.index)
            preds = pd.Series(inverse_log(preds, use_log), index=preds.index)
            return preds

        return ForecastArtifacts(
            train=_predict(train_series),
            val=_predict(val_series),
            test=_predict(test_series),
        )

    @staticmethod
    def _collect_metrics(
        train_series: pd.Series,
        val_series: pd.Series,
        test_series: pd.Series,
        forecasts: ForecastArtifacts,
    ) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}
        if not train_series.empty:
            metrics["Train"] = metric_dict(train_series, forecasts.train.reindex(train_series.index))
        if not val_series.empty:
            metrics["Val"] = metric_dict(val_series, forecasts.val.reindex(val_series.index))
        if not test_series.empty:
            metrics["Test"] = metric_dict(test_series, forecasts.test.reindex(test_series.index))
        return metrics
