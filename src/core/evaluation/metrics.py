from __future__ import annotations

from typing import Iterable

import numpy as np


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean absolute error."""
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Root mean squared error."""
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean absolute percentage error with zero-target protection."""
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)

    denominator = np.where(true == 0, np.nan, true)
    percentage_errors = np.abs((true - pred) / denominator) * 100.0
    valid = ~np.isnan(percentage_errors)

    if not np.any(valid):
        return float("nan")

    return float(np.mean(percentage_errors[valid]))


def wape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Weighted absolute percentage error."""
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)

    denominator = np.sum(np.abs(true))
    if denominator == 0:
        return float("nan")

    return float(np.sum(np.abs(true - pred)) / denominator)


def smape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Symmetric mean absolute percentage error."""
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)

    denominator = np.abs(true) + np.abs(pred)
    denominator = np.where(denominator == 0, 1e-12, denominator)
    return float(np.mean(2.0 * np.abs(pred - true) / denominator))


def metric_dict(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Return a dictionary of key regression metrics."""
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)

    mae_ = float(np.mean(np.abs(true - pred)))
    rmse_ = float(np.sqrt(np.mean((true - pred) ** 2)))
    mape_ = float(
        np.mean(
            np.where(true != 0, np.abs((true - pred) / true), 0.0)
        )
    ) * 100.0

    return {
        "MAE": mae_,
        "RMSE": rmse_,
        "MAPE": mape_,
        "sMAPE": smape(true, pred),
        "WAPE": wape(true, pred),
    }
