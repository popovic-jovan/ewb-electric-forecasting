"""Abstract model interface used by the unified training CLI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd


@dataclass
class ModelInfo:
    """Metadata describing a registered model implementation."""

    name: str
    display_name: str
    default_train_config: Optional[Path] = None
    default_tune_config: Optional[Path] = None
    description: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class TrainResult:
    """Container for outputs returned by a model training run."""

    fitted_model: Any
    metrics: Mapping[str, float]
    artifacts: Dict[str, Path]
    model_path: Optional[Path] = None


class ModelBase(ABC):
    """Common contract implemented by every supported model family."""

    info: ModelInfo

    def __init__(self, config: Mapping[str, Any], dataset_config: Mapping[str, Any]):
        self.config = dict(config)
        self.dataset_config = dict(dataset_config)

    # --------------------------------------------------------------------- #
    # Training / tuning
    # --------------------------------------------------------------------- #
    @abstractmethod
    def train(self, data: pd.DataFrame, output_dir: Path) -> TrainResult:
        """Fit the model on the provided dataframe and persist artifacts."""

    def tune(self, data: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
        """Optional hyperparameter tuning hook."""
        raise NotImplementedError(f"{self.info.name} does not implement tuning.")

    # --------------------------------------------------------------------- #
    # Prediction / inference
    # --------------------------------------------------------------------- #
    @abstractmethod
    def predict(
        self,
        model_path: Path,
        data: pd.DataFrame,
        horizon: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate forecasts using a persisted model artifact."""

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #
    def save_artifact(self, artifact: Any, path: Path) -> Path:
        """Persist an artifact to disk and return the resolved path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(artifact, "to_pickle"):
            artifact.to_pickle(path)
        elif hasattr(artifact, "save"):
            artifact.save(path)
        else:
            import pickle

            with path.open("wb") as handle:
                pickle.dump(artifact, handle)
        return path.resolve()

    def build_output_dir(self, root: Path, *parts: str) -> Path:
        """Construct (and optionally create) a nested output directory path."""
        path = root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

