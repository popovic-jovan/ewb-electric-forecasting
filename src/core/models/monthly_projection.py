"""Monthly projection baseline that scales observed usage to full-month totals."""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from src.core.data.preperation import (
    load_aggregated_series,
    load_raw_series,
    split_by_time_markers,
)
from src.core.evaluation.metrics import metric_dict
from src.core.models import ModelBase, ModelInfo, TrainResult
from src.core.registry import register


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    """Ensure timestamps are timezone-naive for consistent monthly math."""
    ts = pd.to_datetime(series, errors="coerce")
    tz = getattr(ts.dt, "tz", None)
    if tz is not None:
        ts = ts.dt.tz_convert(None)
    return ts


@dataclass
class _MonthInfo:
    actual_total: float
    total_days: int
    month_start: pd.Timestamp


@register
class MonthlyProjectionModel(ModelBase):
    """Project month-end usage via average consumption-to-date."""

    info = ModelInfo(
        name="monthly_projection",
        display_name="Monthly Projection Baseline",
        default_train_config=Path("configs/model/monthly_projection.yaml"),
        default_tune_config=None,
        description=(
            "Scales the observed average daily usage in a month to the calendar length "
            "((usage_so_far / days_so_far) * total_days_in_month)."
        ),
        tags=("baseline", "time-series"),
    )

    def __init__(self, config: Mapping[str, object], dataset_config: Mapping[str, object]):
        super().__init__(config, dataset_config)
        observation_cfg = dict(self.config.get("observation", {}))
        self.min_days_observed = int(observation_cfg.get("min_days", 3))
        max_days = observation_cfg.get("max_days")
        self.max_days_observed = int(max_days) if isinstance(max_days, (int, float)) and max_days else None
        self.require_month_start = bool(observation_cfg.get("require_month_start", True))
        self.min_actual_coverage = float(self.config.get("min_actual_coverage", 0.95))
        self.dataset_mode = str(self.config.get("dataset_mode", "aggregated")).lower()
        if self.dataset_mode not in {"aggregated", "raw"}:
            raise ValueError("monthly_projection.dataset_mode must be 'aggregated' or 'raw'.")

    # ------------------------------------------------------------------ #

    def train(self, data: pd.DataFrame, output_dir: Path) -> TrainResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = output_dir / "reports"
        predictions_dir = output_dir / "predictions"
        reports_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)

        target_col = self.dataset_config.get("target_col", "DELIVERED_VALUE")
        if self.dataset_mode == "raw":
            group_col = self.dataset_config.get("id_col")
            raw_df = load_raw_series(self.dataset_config, data)
            if raw_df.empty:
                raise ValueError("Monthly projection baseline requires non-empty data.")
            if not group_col or group_col not in raw_df.columns:
                raise ValueError("Raw dataset mode requires a valid 'id_col' present in the dataframe.")
            source_df = raw_df
        else:
            group_col = None
            source_df = load_aggregated_series(self.dataset_config, data)
            if source_df.empty:
                raise ValueError("Monthly projection baseline requires non-empty aggregated data.")

        month_lookup = self._build_month_lookup(source_df, target_col, group_col)
        if not month_lookup:
            raise RuntimeError(
                "No months met the coverage threshold; consider lowering 'min_actual_coverage'."
            )

        frames = split_by_time_markers(source_df, self.dataset_config)
        split_map = {"Train": frames.train, "Val": frames.val, "Test": frames.test}
        monthly_frames: list[pd.DataFrame] = []
        metrics_summary: Dict[str, dict[str, float]] = {}

        for split_name, frame in split_map.items():
            summary = self._summarize_split(frame, month_lookup, target_col, group_col)
            if summary.empty:
                continue
            summary.insert(0, "split", split_name)
            monthly_frames.append(summary)
            metrics = metric_dict(summary["actual_total"], summary["projected_total"])
            metrics["n_months"] = float(len(summary))
            metrics["avg_days_observed"] = float(summary["days_observed"].mean())
            metrics_summary[split_name] = metrics

        metrics_path = reports_dir / "metrics.csv"
        metrics_records = [
            {"split": split, **values} for split, values in metrics_summary.items()
        ]
        pd.DataFrame(metrics_records).to_csv(metrics_path, index=False)

        predictions_path = predictions_dir / "monthly_projection.csv"
        if monthly_frames:
            pd.concat(monthly_frames, ignore_index=True).to_csv(predictions_path, index=False)
        else:
            columns = ["split"]
            if group_col:
                columns.append(group_col)
            columns.extend(
                [
                    "month",
                    "days_observed",
                    "total_days",
                    "energy_so_far",
                    "projected_total",
                    "actual_total",
                    "observation_end",
                ]
            )
            pd.DataFrame(columns=columns).to_csv(predictions_path, index=False)

        artifacts = {
            "reports_dir": reports_dir,
            "predictions": predictions_path,
            "metrics": metrics_path,
        }

        summary_metrics = metrics_summary.get(
            "Test",
            metrics_summary.get("Val", metrics_summary.get("Train", {})),
        )
        return TrainResult(
            fitted_model=None,
            metrics=summary_metrics,
            artifacts=artifacts,
            model_path=None,
        )

    def tune(self, data: pd.DataFrame, output_dir: Path) -> dict[str, float]:
        raise NotImplementedError("Monthly projection baseline does not support tuning.")

    def predict(
        self,
        model_path: Path,
        data: pd.DataFrame,
        horizon: int | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError("Use training reports for the projection baseline.")

    # ------------------------------------------------------------------ #

    def _build_month_lookup(
        self,
        df: pd.DataFrame,
        target_col: str,
        group_col: str | None = None,
    ) -> Dict[tuple[object, pd.Period], _MonthInfo]:
        frame = df.dropna(subset=["timestamp"]).copy()
        if frame.empty:
            return {}
        frame["timestamp"] = _normalize_timestamp(frame["timestamp"])
        frame["day"] = frame["timestamp"].dt.floor("D")
        frame["month"] = frame["timestamp"].dt.to_period("M")
        group_fields: list[str] = ["month"]
        if group_col:
            group_fields.insert(0, group_col)
        grouped = frame.groupby(group_fields).agg(
            actual_total=(target_col, "sum"),
            days_observed=("day", "nunique"),
            month_start=("day", "min"),
        ).reset_index()
        grouped["total_days"] = grouped["month"].apply(lambda period: monthrange(period.year, period.month)[1])
        grouped["coverage"] = grouped["days_observed"] / grouped["total_days"]
        eligible = grouped[grouped["coverage"] >= self.min_actual_coverage]

        lookup: Dict[tuple[object, pd.Period], _MonthInfo] = {}
        for _, row in eligible.iterrows():
            period: pd.Period = row["month"]
            month_start = pd.Timestamp(row["month_start"])
            group_value = row[group_col] if group_col else None
            lookup[(group_value, period)] = _MonthInfo(
                actual_total=float(row["actual_total"]),
                total_days=int(row["total_days"]),
                month_start=month_start.normalize(),
            )
        return lookup

    def _summarize_split(
        self,
        frame: pd.DataFrame | None,
        month_lookup: Dict[tuple[object, pd.Period], _MonthInfo],
        target_col: str,
        group_col: str | None = None,
    ) -> pd.DataFrame:
        if frame is None or frame.empty or not month_lookup:
            return pd.DataFrame()

        df = frame.copy()
        df["timestamp"] = _normalize_timestamp(df["timestamp"])
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            return pd.DataFrame()

        df["day"] = df["timestamp"].dt.floor("D")
        df["month"] = df["timestamp"].dt.to_period("M")

        if group_col and group_col in df.columns:
            entity_groups = df.groupby(group_col)
        else:
            entity_groups = [(None, df)]

        records = []
        for entity, entity_df in entity_groups:
            for period, group in entity_df.groupby("month"):
                month_info = month_lookup.get((entity, period))
                if month_info is None:
                    continue
                if self.require_month_start:
                    earliest_day = group["day"].min()
                    if pd.Timestamp(earliest_day).normalize() > month_info.month_start:
                        continue

                unique_days = np.sort(group["day"].unique())
                if len(unique_days) < self.min_days_observed:
                    continue

                if self.max_days_observed is not None and self.max_days_observed < len(unique_days):
                    cutoff = unique_days[self.max_days_observed - 1]
                    observed = group[group["day"] <= cutoff]
                else:
                    observed = group

                days_observed = observed["day"].nunique()
                if days_observed < self.min_days_observed:
                    continue

                energy_so_far = observed[target_col].sum()
                if not np.isfinite(energy_so_far):
                    continue

                projection = float((energy_so_far / days_observed) * month_info.total_days)
                record: Dict[str, object] = {}
                if group_col:
                    record[group_col] = entity
                record.update(
                    {
                        "month": period.strftime("%Y-%m"),
                        "days_observed": int(days_observed),
                        "total_days": int(month_info.total_days),
                        "energy_so_far": float(energy_so_far),
                        "projected_total": projection,
                        "actual_total": month_info.actual_total,
                        "observation_end": observed["timestamp"].max(),
                    }
                )
                records.append(record)

        if not records:
            return pd.DataFrame()
        columns = []
        if group_col:
            columns.append(group_col)
        columns.extend(
            [
                "month",
                "days_observed",
                "total_days",
                "energy_so_far",
                "projected_total",
                "actual_total",
                "observation_end",
            ]
        )
        return pd.DataFrame.from_records(records, columns=columns)
