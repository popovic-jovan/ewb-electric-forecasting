"""Unified command-line interface for training, tuning, and forecasting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.core.pipelines.predict import PredictArgs, run_predict
from src.core.pipelines.train import TrainArgs, run_train
from src.core.pipelines.tune import TuneArgs, run_tune
from src.core.registry import available_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, tune, and predict with energy forecasting models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _build_train_parser(subparsers)
    _build_tune_parser(subparsers)
    _build_predict_parser(subparsers)
    _build_list_parser(subparsers)

    return parser


def _build_train_parser(subparsers: argparse._SubParsersAction) -> None:
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model on the configured dataset.",
    )
    train_parser.add_argument("--model", required=True, help="Model identifier (e.g., sarimax, xgboost).")
    train_parser.add_argument(
        "--dataset-config",
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML.",
    )
    train_parser.add_argument(
        "--experiment-config",
        default="configs/experiment.yaml",
        help="Path to experiment configuration YAML.",
    )
    train_parser.add_argument(
        "--model-config",
        help="Optional path to a model-specific configuration YAML.",
    )
    train_parser.add_argument(
        "--output-root",
        default="models",
        help="Directory where artifacts, reports, and tuning outputs will be stored.",
    )
    train_parser.add_argument(
        "--meter",
        dest="meters",
        action="append",
        help="Restrict training to a specific meter (repeat for multiple).",
    )
    train_parser.add_argument(
        "--quick",
        action="store_true",
        help="Enable a reduced training configuration for faster iteration.",
    )
    train_parser.set_defaults(func=_cmd_train)


def _build_tune_parser(subparsers: argparse._SubParsersAction) -> None:
    tune_parser = subparsers.add_parser(
        "tune",
        help="Hyperparameter tuning for a registered model.",
    )
    tune_parser.add_argument("--model", required=True, help="Model identifier (e.g., sarimax, xgboost).")
    tune_parser.add_argument(
        "--dataset-config",
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML.",
    )
    tune_parser.add_argument(
        "--tune-config",
        help="Path to tuning configuration YAML.",
    )
    tune_parser.add_argument(
        "--model-config",
        help="Optional path to a model configuration YAML (overrides default).",
    )
    tune_parser.add_argument(
        "--output-root",
        default="models",
        help="Directory where tuning studies will be saved.",
    )
    tune_parser.add_argument(
        "--quick",
        action="store_true",
        help="Enable a reduced search space for faster tuning.",
    )
    tune_parser.set_defaults(func=_cmd_tune)


def _build_predict_parser(subparsers: argparse._SubParsersAction) -> None:
    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate forecasts from a persisted model checkpoint.",
    )
    predict_parser.add_argument("--model", required=True, help="Model identifier.")
    predict_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained model artifact to load.",
    )
    predict_parser.add_argument(
        "--dataset-config",
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML.",
    )
    predict_parser.add_argument(
        "--model-config",
        help="Optional path to a model configuration YAML.",
    )
    predict_parser.add_argument(
        "--data",
        help="Optional CSV containing recent observations to forecast.",
    )
    predict_parser.add_argument(
        "--output",
        help="Optional path to save predictions as CSV. Defaults to stdout preview.",
    )
    predict_parser.set_defaults(func=_cmd_predict)


def _build_list_parser(subparsers: argparse._SubParsersAction) -> None:
    list_parser = subparsers.add_parser(
        "list-models",
        help="Show registered models and default configurations.",
    )
    list_parser.set_defaults(func=_cmd_list_models)


def _cmd_train(args: argparse.Namespace) -> int:
    meters: Sequence[str] | None = args.meters
    train_args = TrainArgs(
        model=args.model,
        dataset_config=Path(args.dataset_config),
        experiment_config=Path(args.experiment_config) if args.experiment_config else None,
        model_config=Path(args.model_config) if args.model_config else None,
        output_root=Path(args.output_root),
        meters=meters,
        quick=bool(getattr(args, "quick", False)),
    )
    if train_args.quick:
        print("Quick mode enabled: using reduced training configuration.")
    result = run_train(train_args)

    if result.metrics:
        print("Training metrics:")
        for key, value in sorted(result.metrics.items()):
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    if result.artifacts:
        print("Artifacts:")
        for name, path in result.artifacts.items():
            print(f"  {name}: {path}")

    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    tune_args = TuneArgs(
        model=args.model,
        dataset_config=Path(args.dataset_config),
        tune_config=Path(args.tune_config) if args.tune_config else None,
        model_config=Path(args.model_config) if args.model_config else None,
        output_root=Path(args.output_root),
        quick=bool(getattr(args, "quick", False)),
    )
    metrics = run_tune(tune_args)
    print("Tuning summary:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    predict_args = PredictArgs(
        model=args.model,
        checkpoint=Path(args.checkpoint),
        dataset_config=Path(args.dataset_config),
        model_config=Path(args.model_config) if args.model_config else None,
        data_path=Path(args.data) if args.data else None,
    )
    df = run_predict(predict_args)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    else:
        print("Predictions preview:")
        print(df.head())
    return 0


def _cmd_list_models(_: argparse.Namespace) -> int:
    print("Registered models:")
    infos = list(available_models())
    for info in infos:
        default_cfg = f"(train cfg: {info.default_train_config})" if info.default_train_config else ""
        print(f"  - {info.name}: {info.display_name} {default_cfg}")
    if not infos:
        print("  (no models registered yet)")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if hasattr(args, "func"):
        return args.func(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
