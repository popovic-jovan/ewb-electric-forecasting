"""Model implementations registered for training/tuning/prediction."""

from __future__ import annotations

from .base import ModelBase, ModelInfo, TrainResult
from .xgb_wrapper import XGBModel

__all__ = [
    "ModelBase",
    "ModelInfo",
    "TrainResult",
    "XGBModel",
]

