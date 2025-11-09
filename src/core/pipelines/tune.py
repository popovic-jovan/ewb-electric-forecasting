"""Unified tuning pipeline entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from src.core.io import load_yaml
from src.core.registry import get as get_model


@dataclass
class TuneArgs:
    model: str
    dataset_config: Path = Path("configs/dataset.yaml")
    tune_config: Path | None = None
    model_config: Path | None = None
    output_root: Path = Path("models")
    quick: bool = False


def run_tune(args: TuneArgs) -> Mapping[str, float]:
    dataset_cfg = load_yaml(args.dataset_config)
    model_cfg: Mapping[str, object] = {}
    if args.model_config and Path(args.model_config).exists():
        model_cfg = load_yaml(args.model_config)
    elif args.tune_config and Path(args.tune_config).exists():
        model_cfg = load_yaml(args.tune_config)
    model_cfg = dict(model_cfg or {})
    if args.quick:
        model_cfg["_runtime_quick"] = True

    model_cls = get_model(args.model)
    output_dir = args.output_root / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_cfg["raw_csv"])
    model = model_cls(model_cfg, dataset_cfg)
    return model.tune(df, output_dir)
