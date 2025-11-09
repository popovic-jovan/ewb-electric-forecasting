"""Batch-train SARIMA daily models for every meter (except exclusions)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.core.io import load_yaml
from src.core.pipelines.train import TrainArgs, run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SARIMA daily models for all meters.")
    parser.add_argument(
        "--dataset-config",
        default="configs/dataset_daily.yaml",
        help="Path to dataset configuration YAML.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/sarima_daily.yaml",
        help="Path to SARIMA model configuration YAML.",
    )
    parser.add_argument(
        "--output-root",
        default="models",
        help="Directory where per-meter artifacts will be stored.",
    )
    parser.add_argument(
        "--exclude",
        default=["34"],
        nargs="*",
        help="Meter identifiers to skip (default excludes meter 34).",
    )
    return parser.parse_args()


def _list_meters(dataset_cfg_path: Path, exclude: Sequence[str]) -> list[str]:
    dataset_cfg = load_yaml(dataset_cfg_path)
    id_col = dataset_cfg.get("id_col")
    if not id_col:
        raise ValueError("Dataset configuration must define 'id_col'.")

    df = pd.read_csv(dataset_cfg["raw_csv"], usecols=[id_col])
    unique_ids = sorted(df[id_col].astype(str).unique())
    exclude_set = {str(x) for x in exclude}
    return [mid for mid in unique_ids if mid not in exclude_set]


def train_all(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset_config)
    model_path = Path(args.model_config)
    output_root = Path(args.output_root)

    meters = _list_meters(dataset_path, args.exclude)
    print(f"Training SARIMA daily for {len(meters)} meters...")

    metrics: dict[str, float] = {}
    for meter in meters:
        print(f"\n=== Meter {meter} ===")
        train_args = TrainArgs(
            model="sarima",
            dataset_config=dataset_path,
            model_config=model_path,
            output_root=output_root,
            meters=[meter],
        )
        try:
            result = run_train(train_args)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Meter {meter} failed: {exc}")
            continue
        wape_value = result.metrics.get("WAPE")
        if wape_value is not None:
            metrics[meter] = float(wape_value)
            print(f"Meter {meter} WAPE: {wape_value:.4f}")
        else:
            print(f"Meter {meter} did not report WAPE.")

    if not metrics:
        print("No successful runs to summarize.")
        return

    print("\n=== Summary ===")
    for meter, value in sorted(metrics.items()):
        print(f"{meter}: WAPE={value:.4f}")
    avg_wape = sum(metrics.values()) / len(metrics)
    print(f"\nAverage WAPE across {len(metrics)} meters: {avg_wape:.4f}")


if __name__ == "__main__":
    train_all(parse_args())
