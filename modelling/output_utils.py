"""Utilities for managing model output directories."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


def _sanitize(name: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in name.strip())
    clean = clean.strip("-_")
    return clean or "run"


def prepare_run_directory(
    base_dir: Path | str,
    run_name: Optional[str] = None,
    *,
    timestamp: bool = True,
) -> Path:
    """Create and return a unique directory for a model run.

    Parameters
    ----------
    base_dir:
        Root directory where runs should be stored. It is created if missing.
    run_name:
        Optional human-friendly label (e.g., "experiment_a").
    timestamp:
        When True (default) append a ``YYYYMMDD-HHMMSS`` suffix so repeated runs
        do not overwrite each other.
    """

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    parts = []
    if run_name:
        parts.append(_sanitize(run_name))
    if timestamp:
        parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))

    if parts:
        folder = "_".join(parts)
    else:
        folder = "run"

    run_path = base_path / folder
    counter = 1
    while run_path.exists():
        counter += 1
        run_path = base_path / f"{folder}_{counter:02d}"

    run_path.mkdir(parents=True, exist_ok=False)
    return run_path
