"""Runtime registry that maps model identifiers to their implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Iterable, Mapping, Type

from .models import ModelBase, ModelInfo

REGISTRY: Dict[str, Type[ModelBase]] = {}
METADATA: Dict[str, ModelInfo] = {}
DEFAULT_MODULES: tuple[str, ...] = (
    "seasonal_naive",
    "sarimax",
    "xgboost",
    "sarima",
    "prophet",
    "lstm",
    "monthly_projection",
)


def register(model_cls: Type[ModelBase]) -> Type[ModelBase]:
    """Register a ModelBase subclass for CLI/UI discovery."""
    if not hasattr(model_cls, "info"):
        raise AttributeError(f"{model_cls.__name__} is missing 'info' metadata.")

    info = model_cls.info
    key = info.name.lower()
    if key in REGISTRY:
        raise ValueError(f"Model '{key}' already registered.")

    REGISTRY[key] = model_cls
    METADATA[key] = info
    return model_cls


def get(model_name: str) -> Type[ModelBase]:
    """Return the model class registered under the provided name."""
    key = model_name.lower()
    if key not in REGISTRY:
        _lazy_import(key)
    try:
        return REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Model '{model_name}' is not registered.") from exc


def available_models() -> Iterable[ModelInfo]:
    """Iterate over metadata describing all known models."""
    for module_name in DEFAULT_MODULES:
        _lazy_import(module_name)
    for key in sorted(METADATA):
        yield METADATA[key]


def _lazy_import(model_name: str) -> None:
    """Attempt to import a module following naming convention core.models.<name>."""
    module_name = f"src.core.models.{model_name}"
    try:
        import_module(module_name)
    except ModuleNotFoundError:
        # Swallow errors; actual lookup will raise a clearer KeyError later.
        return
