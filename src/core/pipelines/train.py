"""Unified training pipeline entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from src.core.io import load_yaml
from src.core.models.base import TrainResult
from src.core.registry import get as get_model


@dataclass
class TrainArgs:
    model: str
    dataset_config: Path = Path("configs/dataset.yaml")
    experiment_config: Path | None = Path("configs/experiment.yaml")
    model_config: Path | None = None
    output_root: Path = Path("models")
    meters: Sequence[str] | None = None
    quick: bool = False


def run_train(args: TrainArgs) -> TrainResult:
    """Dispatch training to a registered model."""
    dataset_cfg = dict(load_yaml(args.dataset_config))
    experiment_cfg: Mapping[str, object] = {}
    if args.experiment_config and Path(args.experiment_config).exists():
        experiment_cfg = load_yaml(args.experiment_config)
    model_cfg: Mapping[str, object] = {}
    if args.model_config and Path(args.model_config).exists():
        model_cfg = load_yaml(args.model_config)
    if not model_cfg and experiment_cfg:
        model_section = experiment_cfg.get(args.model)
        if isinstance(model_section, Mapping):
            cfg_path = model_section.get("model_cfg_path")
            if cfg_path:
                cfg_path = Path(cfg_path)
                if cfg_path.exists():
                    model_cfg = load_yaml(cfg_path)
    model_cfg = dict(model_cfg) if model_cfg else {}
    if args.quick:
        model_cfg["_runtime_quick"] = True

    model_cls = get_model(args.model)
    df = pd.read_csv(dataset_cfg["raw_csv"])
    selected_meters: list[str] = []
    if args.meters:
        id_col = dataset_cfg.get("id_col")
        if not id_col:
            raise ValueError("Dataset config must define 'id_col' to use --meter filtering.")
        selected_meters = [str(meter) for meter in args.meters]
        df[id_col] = df[id_col].astype(str)
        df = df[df[id_col].isin(selected_meters)]
        if df.empty:
            raise ValueError(f"No rows remain after filtering to meters {selected_meters}.")
        model_cfg["_selected_meters"] = selected_meters
        dataset_cfg["_selected_meters"] = selected_meters

    output_root = args.output_root / args.model
    if len(selected_meters) == 1:
        output_root = output_root / f"meter_{selected_meters[0]}"
    output_root.mkdir(parents=True, exist_ok=True)

    model = model_cls(model_cfg, dataset_cfg)
    return model.train(df, output_root)
