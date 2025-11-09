import pandas as pd

from src.core.models.monthly_projection import MonthlyProjectionModel


def test_monthly_projection_baseline(tmp_path):
    """Monthly projection should aggregate months and emit predictions."""
    hours = 24 * (31 + 28 + 31)
    timestamps = pd.date_range("2023-01-01", periods=hours, freq="h")
    df = pd.DataFrame(
        {
            "METER_UI": ["meter_1"] * hours,
            "AGGREGATE_DATE": timestamps,
            "DELIVERED_VALUE": timestamps.day.astype(float),
            "Daily Energy Usage": timestamps.day.astype(float),
            "RECEIVED_VALUE": 0.0,
            "Quarter": timestamps.quarter,
            "power_zero": 0,
            "daily_energy_zero": 0,
            "Ref": range(hours),
            "Row": range(hours),
            "AGGREGATE_YEAR": timestamps.year,
            "AGGREGATE_MONTH": timestamps.month,
            "AGGREGATE_DAY": timestamps.day,
            "Error Check day": 0,
            "AGGREGATE_HOUR": timestamps.hour,
            "Error Check Hour": 0,
        }
    )

    dataset_cfg = {
        "time_col": "AGGREGATE_DATE",
        "target_col": "DELIVERED_VALUE",
        "freq": "h",
        "train_end": "2023-01-31 23:00:00",
        "val_end": "2023-02-28 23:00:00",
        "test_end": "2023-03-31 23:00:00",
    }
    model_cfg = {
        "observation": {"min_days": 5, "max_days": 10, "require_month_start": True},
        "min_actual_coverage": 1.0,
    }

    model = MonthlyProjectionModel(model_cfg, dataset_cfg)
    result = model.train(df, tmp_path / "monthly_projection")

    assert result.metrics, "Expected summary metrics from the latest split."
    assert "MAE" in result.metrics

    metrics_path = result.artifacts["metrics"]
    preds_path = result.artifacts["predictions"]
    assert metrics_path.exists()
    assert preds_path.exists()

    preds_df = pd.read_csv(preds_path)
    assert {"split", "month", "projected_total", "actual_total"}.issubset(preds_df.columns)
    assert not preds_df.empty

    march = preds_df[(preds_df["split"] == "Test") & (preds_df["month"] == "2023-03")]
    assert not march.empty
    # Projection should differ from the actual since we limit observations to 10 days.
    assert (march["projected_total"] != march["actual_total"]).all()


def test_monthly_projection_raw_mode(tmp_path):
    """Monthly projection should support per-meter raw datasets."""
    hours = 24 * (31 + 28 + 31)
    timestamps = pd.date_range("2023-01-01", periods=hours, freq="h")
    base_values = timestamps.day.astype(float)

    def _build_meter_frame(meter_id: str, scale: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "METER_UI": [meter_id] * hours,
                "AGGREGATE_DATE": timestamps,
                "DELIVERED_VALUE": base_values * scale,
                "Daily Energy Usage": base_values * scale,
                "RECEIVED_VALUE": 0.0,
                "Quarter": timestamps.quarter,
                "power_zero": 0,
                "daily_energy_zero": 0,
                "Ref": range(hours),
                "Row": range(hours),
                "AGGREGATE_YEAR": timestamps.year,
                "AGGREGATE_MONTH": timestamps.month,
                "AGGREGATE_DAY": timestamps.day,
                "Error Check day": 0,
                "AGGREGATE_HOUR": timestamps.hour,
                "Error Check Hour": 0,
            }
        )

    df = pd.concat(
        [
            _build_meter_frame("meter_A", 1.0),
            _build_meter_frame("meter_B", 0.6),
        ],
        ignore_index=True,
    )

    dataset_cfg = {
        "time_col": "AGGREGATE_DATE",
        "target_col": "DELIVERED_VALUE",
        "id_col": "METER_UI",
        "freq": "h",
        "train_end": "2023-01-31 23:00:00",
        "val_end": "2023-02-28 23:00:00",
        "test_end": "2023-03-31 23:00:00",
    }
    model_cfg = {
        "dataset_mode": "raw",
        "observation": {"min_days": 5, "max_days": 8, "require_month_start": True},
        "min_actual_coverage": 1.0,
    }

    model = MonthlyProjectionModel(model_cfg, dataset_cfg)
    result = model.train(df, tmp_path / "monthly_projection_raw")

    assert result.metrics and "MAE" in result.metrics

    preds_df = pd.read_csv(result.artifacts["predictions"])
    assert "METER_UI" in preds_df.columns
    assert set(preds_df["split"]) <= {"Train", "Val", "Test"}
    assert preds_df["METER_UI"].nunique() == 2
