import json

import numpy as np
import pandas as pd

from src.core.evaluation.metrics import mae
from src.core.models import XGBModel
from src.core.models.xgboost import XGBoostAggregatedModel


def test_training_pipeline_smoke(tmp_path):
    """Ensure the core XGBoost pipeline trains and produces finite metrics."""
    periods = 72
    timestamps = pd.date_range("2023-01-01", periods=periods, freq="h")
    signal = np.sin(np.linspace(0, 6 * np.pi, periods)) + 5
    noise = np.random.default_rng(seed=123).normal(0, 0.1, periods)

    df = pd.DataFrame(
        {
            "METER_UI": ["meter_1"] * periods,
            "AGGREGATE_DATE": timestamps,
            "DELIVERED_VALUE": signal + noise,
            "Daily Energy Usage": signal + 1,
            "RECEIVED_VALUE": np.maximum(0, signal - 5),
            "Quarter": timestamps.quarter,
            "power_zero": (np.arange(periods) % 12 == 0).astype(int),
            "daily_energy_zero": 0,
            "Ref": np.arange(periods),
            "Row": np.arange(periods),
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
        "id_col": "METER_UI",
        "freq": "h",
        "train_end": "2023-01-02 23:00:00",
        "val_end": "2023-01-03 11:00:00",
        "test_end": "2023-01-03 23:00:00",
    }

    model_cfg = {
        "params": {
            "n_estimators": 25,
            "max_depth": 3,
            "learning_rate": 0.2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
        },
        "fit": {"early_stopping_rounds": 5},
    }

    model = XGBoostAggregatedModel(model_cfg, dataset_cfg)
    result = model.train(df, tmp_path / "outputs")

    assert result.metrics, "Expected the training run to report metrics."
    assert np.isfinite(result.metrics.get("MAE", np.nan))
    assert result.model_path and result.model_path.exists()

    wrapped = XGBModel.load(result.model_path)
    predictions_dir = result.artifacts["predictions_dir"]
    test_preds_path = predictions_dir / "test.csv"
    assert test_preds_path.exists()

    preds_df = pd.read_csv(test_preds_path)
    if not preds_df.empty:
        assert np.isfinite(mae(preds_df["DELIVERED_VALUE"], preds_df["y_hat"]))

    (
        target_col,
        feature_cols,
        _train_df,
        _val_df,
        test_df,
        ts_col,
        group_col,
    ) = model._prepare_datasets(df)
    assert target_col == "DELIVERED_VALUE"
    assert ts_col == "timestamp"
    assert group_col is None
    assert feature_cols, "Feature column list should not be empty."

    with result.artifacts["feature_columns"].open("r", encoding="utf-8") as handle:
        persisted_feature_cols = json.load(handle)
    assert persisted_feature_cols == feature_cols

    sample_features = test_df[feature_cols].head(1)

    if not sample_features.empty:
        wrapped.predict(sample_features)
