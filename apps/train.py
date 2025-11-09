"""Legacy training shim maintained for backwards compatibility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from apps.cli import main as cli_main  # noqa: E402


def main() -> None:
    """Parse legacy arguments and forward to the unified CLI."""
    parser = argparse.ArgumentParser(description="Train or tune forecasting models (legacy wrapper).")
    parser.add_argument(
        "--mode",
        choices=["train", "tune-grid"],
        default="train",
        help="Use 'train' for a standard run or 'tune-grid' for the historical grid search.",
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "sarimax", "lstm"],
        default="xgboost",
        help="Model identifier. Additional models are available via apps/cli.py.",
    )
    parser.add_argument(
        "--meter",
        dest="meters",
        action="append",
        help="Restrict SARIMAX training to a specific meter (repeatable).",
    )
    args = parser.parse_args()

    if args.mode == "tune-grid":
        tune_args = ["tune", "--model", args.model]
        exit_code = cli_main(tune_args)
        if exit_code != 0:
            sys.exit(exit_code)
        return

    cli_args = ["train", "--model", args.model]
    if args.meters:
        for meter in args.meters:
            cli_args.extend(["--meter", meter])

    exit_code = cli_main(cli_args)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
