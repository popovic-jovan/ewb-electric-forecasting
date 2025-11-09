"""Aggregated SARIMAX implementation registered with the unified CLI."""

from __future__ import annotations

import itertools
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from core.data.preperation import SplitFrames, load_aggregated_series, load_raw_series, split_by_time_markers
from src.core.evaluation.metrics import metric_dict, wape
from src.core.features.time_series import build_time_series_features
from src.core.models import ModelBase, ModelInfo, TrainResult
from src.core.models.utils import assign_frequency, inverse_log, prepare_feature_splits, prepare_series_and_exog
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
        preferred_orders = cfg.get("quick_orders") or [(1, 1, 1), (1, 1, 0), (2, 1, 1)]
        filtered_orders = []
        for candidate in preferred_orders:
            candidate = tuple(int(x) for x in candidate)
            if candidate in orders and candidate not in filtered_orders:
                filtered_orders.append(candidate)
        if not filtered_orders:
            filtered_orders = orders[: min(3, len(orders))]
        orders = filtered_orders

        preferred_seasonals = cfg.get("quick_seasonal_orders") or [
            (1, 1, 1, seasonal_period),
            (0, 1, 1, seasonal_period),
        ]
        filtered_seasonals = []
        for candidate in preferred_seasonals:
            candidate = (
                int(candidate[0]),
                int(candidate[1]),
                int(candidate[2]),
                int(candidate[3]) if len(candidate) > 3 else seasonal_period,
            )
            if candidate in seasonal_orders and candidate not in filtered_seasonals:
                filtered_seasonals.append(candidate)
        if not filtered_seasonals:
            filtered_seasonals = seasonal_orders[: min(2, len(seasonal_orders))]
        seasonal_orders = filtered_seasonals

    for order in orders:
        for seasonal_order in seasonal_orders:
            yield order, seasonal_order


@dataclass
class ForecastArtifacts:
    train: pd.Series
    val: pd.Series
    test: pd.Series


def fit_sarimax(
    endog: pd.Series,
    exog: pd.DataFrame | None,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    trend: str = "c",
    freq: str | None = None,
):
    """Fit a SARIMAX model and return the fitted results object."""
    dates = None
    resolved_freq = freq

    if isinstance(endog.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        dates = endog.index
        if resolved_freq is None:
            resolved_freq = getattr(endog.index, "freqstr", None)
        if resolved_freq is None and isinstance(endog.index, pd.DatetimeIndex):
            try:
                resolved_freq = pd.infer_freq(endog.index)
            except (TypeError, ValueError):
                resolved_freq = None

    X = None
    if exog is not None and not exog.empty:
        X = exog.select_dtypes(include=[np.number]).astype("float64")
        X = X.loc[:, X.nunique(dropna=False) > 1]
        if X.shape[1] > 1:
            Q, R = np.linalg.qr(X.values)
            keep = np.where(np.abs(np.diag(R)) > 1e-10)[0]
            if len(keep) < X.shape[1]:
                X = X.iloc[:, keep]
        X = X.reindex(endog.index)
        if X.isna().any().any():
            X = X.ffill().bfill()
        if X.empty:
            X = None

    p, d, q = order
    P, D, Q, m = seasonal_order
    safe_trend = "n" if (d + D) > 0 or (X is not None and X.shape[1] >= 1) else trend

    model = SARIMAX(
        endog=endog,
        exog=X,
        order=order,
        seasonal_order=seasonal_order,
        trend=safe_trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
        dates=dates,
        freq=resolved_freq,
        initialization="approximate_diffuse",
        concentrate_scale=True,
    )

    try:
        result = model.fit(method="lbfgs", disp=False, maxiter=200)
    except Exception:
        result = None
        for method in ("powell", "nm", "bfgs"):
            try:
                result = model.fit(method=method, disp=False, maxiter=400)
                break
            except Exception:
                continue
        if result is None:
            raise
    return result


def safe_predict(
    result,
    start,
    end,
    exog: pd.DataFrame | None = None,
) -> pd.Series:
    """Generate predictions aligned to the provided timestamp range."""
    forecast = result.predict(
        start=start,
        end=end,
        exog=exog,
        dynamic=False,
    )
    if isinstance(forecast, pd.Series):
        return forecast

    index = None
    if isinstance(start, Iterable) and not isinstance(start, (str, pd.Timestamp)):
        index = pd.Index(start)
    elif isinstance(end, Iterable) and not isinstance(end, (str, pd.Timestamp)):
        index = pd.Index(end)
    elif hasattr(result.model.data, "row_labels"):
        index = result.model.data.row_labels[start:end]
    else:
        index = pd.RangeIndex(start=0, stop=len(forecast))

    return pd.Series(forecast, index=index)


def serialize_model(result) -> bytes:
    """Serialize a fitted SARIMAX result to bytes."""
    return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_model(payload: bytes):
    """Restore a SARIMAX fitted result from bytes."""
    return pickle.loads(payload)


@register
class SarimaxModel(ModelBase):
    info = ModelInfo(
        name="sarimax",
        display_name="SARIMAX (aggregated)",
        default_train_config=Path("configs/model/sarimax.yaml"),
        default_tune_config=Path("configs/model/sarimax.yaml"),
        description="Seasonal ARIMA with exogenous regressors applied to the aggregated load series.",
        tags=("time-series", "statsmodels"),
    )

    def __init__(self, config: Mapping[str, object], dataset_config: Mapping[str, object]):
        super().__init__(config, dataset_config)
        self._active_meter: str | None = None

    def train(self, data: pd.DataFrame, output_dir: Path) -> TrainResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = output_dir / "artifacts"
        reports_dir = output_dir / "reports"
        predictions_dir = output_dir / "predictions"
        for path in (artifacts_dir, reports_dir, predictions_dir):
            path.mkdir(parents=True, exist_ok=True)

        target_col = self.dataset_config.get("target_col", "DELIVERED_VALUE")
        ts_col = "timestamp"
        use_log = bool(self.config.get("use_log1p", False))
        dataset_mode = str(self.config.get("dataset_mode", "aggregated")).lower()

        group_col: str | None = None
        if dataset_mode == "raw":
            source_df, meter_id = self._load_single_meter_frame(data)
            self._active_meter = meter_id
            group_col = self.dataset_config.get("id_col")
        else:
            source_df = load_aggregated_series(self.dataset_config, data)
            self._active_meter = None

        frames = split_by_time_markers(source_df, self.dataset_config, ts_col=ts_col)

        feature_cfg = self.config.get("features", {})
        train_df, val_df, test_df, exog_cols = prepare_feature_splits(
            frames,
            feature_cfg,
            ts_col,
            target_col,
            group_col=group_col,
        )
        train_df = self._ensure_datetime_index(train_df, ts_col)
        val_df = self._ensure_datetime_index(val_df, ts_col)
        test_df = self._ensure_datetime_index(test_df, ts_col)
        if exog_cols:
            preview = ", ".join(exog_cols[:10])
            suffix = "..." if len(exog_cols) > 10 else ""
            print(f"[SARIMAX] EXOG features ({len(exog_cols)}): {preview}{suffix}")
        else:
            print("[SARIMAX] EXOG features: none (pure SARIMA).")

        freq_hint = assign_frequency([train_df, val_df, test_df], self.dataset_config.get("freq"))

        y_train, X_train = prepare_series_and_exog(train_df, target_col, exog_cols, use_log)
        y_val, X_val = prepare_series_and_exog(val_df, target_col, exog_cols, use_log)
        y_test, X_test = prepare_series_and_exog(test_df, target_col, exog_cols, use_log)

        best_score = math.inf
        best_combo = None
        best_result = None

        raw_candidates = list(_grid_orders(self.config))
        candidates: list[tuple[Tuple[int, int, int], Tuple[int, int, int, int]]] = []
        train_len = len(y_train)
        for order, seasonal_order in raw_candidates:
            m = seasonal_order[3] if len(seasonal_order) > 3 else int(self.config.get("seasonal_period", 24))
            if train_len < 5 * max(1, m) and (seasonal_order[0] + seasonal_order[2]) > 0:
                continue
            candidates.append((order, seasonal_order))
        if not candidates:
            candidates = raw_candidates
        total_candidates = len(candidates)
        for idx, (order, seasonal_order) in enumerate(candidates, start=1):
            print(
                f"[SARIMAX] Grid search {idx}/{total_candidates} evaluating "
                f"order={order} seasonal={seasonal_order}"
            )
            try:
                candidate = fit_sarimax(
                    y_train,
                    X_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=self.config.get("trend", "c"),
                    freq=freq_hint,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[SARIMAX] Failed to fit order={order} seasonal={seasonal_order}: {exc}")
                continue

            if val_df.empty:
                best_combo = (order, seasonal_order)
                best_result = candidate
                break

            try:
                aligned_val_exog = self._align_exog(
                    val_df.index,
                    X_val,
                    getattr(candidate.model, "exog_names", None),
                )
                val_pred = safe_predict(
                    candidate,
                    start=val_df.index[0],
                    end=val_df.index[-1],
                    exog=aligned_val_exog,
                )
                val_pred = pd.Series(val_pred, index=val_df.index)
                val_pred = pd.Series(inverse_log(val_pred, use_log), index=val_df.index)
                val_score = wape(val_df[target_col].values, val_pred.values)
            except Exception as exc:  # noqa: BLE001
                print(f"[SARIMAX] Validation prediction failed order={order} seasonal={seasonal_order}: {exc}")
                continue

            if np.isnan(val_score):
                continue
            if val_score < best_score:
                best_score = val_score
                best_combo = (order, seasonal_order)
                best_result = candidate

        if best_result is None or best_combo is None:
            raise RuntimeError("SARIMAX grid search failed to converge on any configuration.")

        combined_df = pd.concat([train_df, val_df]).sort_index()
        y_combined, X_combined = prepare_series_and_exog(combined_df, target_col, exog_cols, use_log)
        best_result = fit_sarimax(
            y_combined,
            X_combined,
            order=best_combo[0],
            seasonal_order=best_combo[1],
            trend=self.config.get("trend", "c"),
            freq=freq_hint,
        )

        forecasts = self._generate_forecasts(
            best_result,
            train_df,
            val_df,
            test_df,
            X_train,
            X_val,
            X_test,
            use_log,
        )
        metrics = self._collect_metrics(train_df, val_df, test_df, forecasts, target_col)

        metrics_path = reports_dir / "metrics.csv"
        records = [{"split": split, **values} for split, values in metrics.items()]
        if self._active_meter:
            for record in records:
                record["meter"] = self._active_meter
        metrics_df = pd.DataFrame.from_records(records, index=None)
        metrics_df.to_csv(metrics_path, index=False)

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

        model_path = artifacts_dir / "sarimax_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as handle:
            handle.write(serialize_model(best_result))

        return TrainResult(
            fitted_model=None,
            metrics=metrics.get("Test", metrics.get("Val", {})),
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

        dataset_mode = str(self.config.get("dataset_mode", "aggregated")).lower()
        group_col: str | None = None
        if dataset_mode == "raw":
            source_df, meter_id = self._load_single_meter_frame(data)
            self._active_meter = meter_id
            group_col = self.dataset_config.get("id_col")
        else:
            source_df = load_aggregated_series(self.dataset_config, data)
            self._active_meter = None

        frames = split_by_time_markers(source_df, self.dataset_config)
        target_col = self.dataset_config.get("target_col", "DELIVERED_VALUE")

        feature_cfg = self.config.get("features", {})
        train_df, val_df, _, exog_cols = prepare_feature_splits(
            frames,
            feature_cfg,
            "timestamp",
            target_col,
            group_col=group_col,
        )
        train_df = self._ensure_datetime_index(train_df, "timestamp")
        val_df = self._ensure_datetime_index(val_df, "timestamp")

        use_log = bool(self.config.get("use_log1p", False))
        freq_hint = assign_frequency([train_df, val_df], self.dataset_config.get("freq"))

        y_train, X_train = prepare_series_and_exog(train_df, target_col, exog_cols, use_log)
        y_val, X_val = prepare_series_and_exog(val_df, target_col, exog_cols, use_log)

        trials: list[dict[str, object]] = []
        best_combo = None
        best_score = math.inf

        raw_candidates = list(_grid_orders(self.config))
        candidates: list[tuple[Tuple[int, int, int], Tuple[int, int, int, int]]] = []
        train_len = len(y_train)
        for order, seasonal_order in raw_candidates:
            m = seasonal_order[3] if len(seasonal_order) > 3 else int(self.config.get("seasonal_period", 24))
            if train_len < 5 * max(1, m) and (seasonal_order[0] + seasonal_order[2]) > 0:
                continue
            candidates.append((order, seasonal_order))
        if not candidates:
            candidates = raw_candidates
        total_candidates = len(candidates)
        for idx, (order, seasonal_order) in enumerate(candidates, start=1):
            print(
                f"[SARIMAX] Tuning search {idx}/{total_candidates} evaluating "
                f"order={order} seasonal={seasonal_order}"
            )
            try:
                result = fit_sarimax(
                    y_train,
                    X_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=self.config.get("trend", "c"),
                    freq=freq_hint,
                )
                aligned_val_exog = self._align_exog(val_df.index, X_val)
                val_pred = safe_predict(
                    result,
                    start=val_df.index[0],
                    end=val_df.index[-1],
                    exog=aligned_val_exog,
                )
                val_pred = pd.Series(val_pred, index=val_df.index)
                val_pred = pd.Series(inverse_log(val_pred, use_log), index=val_df.index)
                score = wape(val_df[target_col].values, val_pred.values)
            except Exception as exc:  # noqa: BLE001
                print(f"[SARIMAX] Tuning candidate failed ({order}, {seasonal_order}): {exc}")
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

        trials_df = pd.DataFrame(trials)
        trials_path = tuning_dir / "sarimax_trials.csv"
        trials_df.to_csv(trials_path, index=False)

        if best_combo is None:
            raise RuntimeError("Tuning failed to produce a valid SARIMAX configuration.")

        best_path = tuning_dir / "sarimax_best.json"
        best_df = pd.DataFrame(
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
        )
        best_df.to_json(best_path, orient="records", indent=2)
        return {"WAPE": best_score}

    def predict(
        self,
        model_path: Path,
        data: pd.DataFrame,
        horizon: int | None = None,
    ) -> pd.DataFrame:
        if horizon:
            raise NotImplementedError("Future horizon forecasting is not implemented yet for SARIMAX.")
        serialized = model_path.read_bytes()
        result = deserialize_model(serialized)

        aggregated = load_aggregated_series(self.dataset_config, data)
        target_col = self.dataset_config.get("target_col", "DELIVERED_VALUE")
        ts_col = "timestamp"
        feature_cfg = self.config.get("features", {})

        aggregated, exog_cols = build_time_series_features(
            aggregated,
            {"features": feature_cfg},
            ts_col=ts_col,
            target_col=target_col,
        )

        aggregated = aggregated.dropna(subset=[target_col])
        if exog_cols:
            aggregated = aggregated.dropna(subset=exog_cols)

        aggregated = aggregated.sort_values(ts_col).set_index(ts_col)
        aggregated.index = pd.to_datetime(aggregated.index)
        assign_frequency([aggregated], self.dataset_config.get("freq"))

        exog = aggregated[exog_cols] if exog_cols else None
        expected_cols = getattr(result.model, "exog_names", None)
        aligned_exog = self._align_exog(aggregated.index, exog, expected_cols)
        preds = safe_predict(
            result,
            start=aggregated.index[0],
            end=aggregated.index[-1],
            exog=aligned_exog,
        )
        preds = pd.Series(preds, index=aggregated.index)
        if self.config.get("use_log1p", False):
            preds = pd.Series(inverse_log(preds, True), index=preds.index)

        df_out = pd.DataFrame(
            {
                "timestamp": preds.index,
                "y_hat": preds.values,
            }
        )
        if target_col in aggregated.columns:
            df_out["y"] = aggregated[target_col].values
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
                "SARIMAX raw mode expects data for exactly one meter. "
                "Pass --meter to select a single identifier."
            )
        meter_id = unique_ids[0]
        meter_df = df[df[id_col] == meter_id].copy()
        return meter_df, meter_id

    @staticmethod
    def _align_exog(
        index: pd.Index,
        exog: pd.DataFrame | None,
        expected_cols: Sequence[str] | None = None,
    ) -> pd.DataFrame | None:
        if exog is None or exog.empty:
            return None
        aligned = exog.reindex(index)
        if aligned.isna().any().any():
            aligned = aligned.ffill().bfill()
        if expected_cols:
            expected = list(expected_cols)
            aligned = aligned.reindex(columns=expected, fill_value=0.0)
        return aligned.astype("float64")

    @staticmethod
    def _generate_forecasts(
        result,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        X_train: pd.DataFrame | None,
        X_val: pd.DataFrame | None,
        X_test: pd.DataFrame | None,
        use_log: bool,
    ) -> ForecastArtifacts:
        exog_names = getattr(result.model, "exog_names", None)

        def _predict(sub_df: pd.DataFrame, X_subset) -> pd.Series:
            if sub_df.empty:
                return pd.Series(dtype=float)
            aligned = SarimaxModel._align_exog(sub_df.index, X_subset, exog_names)
            preds = safe_predict(
                result,
                start=sub_df.index[0],
                end=sub_df.index[-1],
                exog=aligned,
            )
            preds = pd.Series(preds, index=sub_df.index)
            preds = pd.Series(inverse_log(preds, use_log), index=sub_df.index)
            return preds

        return ForecastArtifacts(
            train=_predict(train_df, X_train),
            val=_predict(val_df, X_val),
            test=_predict(test_df, X_test),
        )

    @staticmethod
    def _collect_metrics(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        forecasts: ForecastArtifacts,
        target_col: str,
    ) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}
        if not train_df.empty:
            metrics["Train"] = metric_dict(train_df[target_col], forecasts.train.reindex(train_df.index))
        if not val_df.empty:
            metrics["Val"] = metric_dict(val_df[target_col], forecasts.val.reindex(val_df.index))
        if not test_df.empty:
            metrics["Test"] = metric_dict(test_df[target_col], forecasts.test.reindex(test_df.index))
        return metrics

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if ts_col not in df.columns:
            raise KeyError(f"Expected timestamp column '{ts_col}' in dataframe.")
        out = df.copy()
        out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
        out = out.dropna(subset=[ts_col])
        out = out.set_index(ts_col, drop=False).sort_index()
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.DatetimeIndex(out.index)
        out.index.name = ts_col
        return out
