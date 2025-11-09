"""Unified prediction pipeline entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

from src.core.registry import get as get_model
from src.core.io import load_yaml


@dataclass
class PredictArgs:
    model: str
    checkpoint: Path
    dataset_config: Path = Path("configs/dataset.yaml")
    model_config: Path | None = None
    data_path: Path | None = None


def run_predict(args: PredictArgs, dataframe: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    dataset_cfg = load_yaml(args.dataset_config)
    model_cfg: Mapping[str, object] = {}
    if args.model_config and Path(args.model_config).exists():
        model_cfg = load_yaml(args.model_config)

    if dataframe is None:
        if args.data_path:
            dataframe = pd.read_csv(args.data_path)
        else:
            dataframe = pd.read_csv(dataset_cfg["raw_csv"])

    model_cls = get_model(args.model)
    model = model_cls(model_cfg, dataset_cfg)
    return model.predict(Path(args.checkpoint), dataframe.copy(), horizon=None)
