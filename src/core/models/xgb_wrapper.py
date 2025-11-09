"""Simple persistence wrapper around XGBoost's regressor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import xgboost as xgb


class XGBModel:
    """Convenience wrapper to standardise usage inside pipelines and serving."""

    def __init__(self, params: Optional[Dict[str, Any]] = None, fit_params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.fit_params = fit_params or {}
        self.model = xgb.XGBRegressor(**self.params)

    def fit(
        self,
        X_train: Any,
        y_train: Iterable[float],
        X_val: Optional[Any] = None,
        y_val: Optional[Iterable[float]] = None,
        **extra_fit_kwargs: Any,
    ) -> None:
        """Train the model, falling back gracefully if validation data is missing."""
        fit_kwargs = dict(self.fit_params)
        eval_metric = fit_kwargs.pop("eval_metric", None)
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
        elif "early_stopping_rounds" in fit_kwargs:
            patience = fit_kwargs.pop("early_stopping_rounds")
            print(
                "Validation data unavailable; skipping early stopping "
                f"(early_stopping_rounds={patience} removed)."
            )

        early_stopping_rounds = fit_kwargs.pop("early_stopping_rounds", None)

        if eval_set and early_stopping_rounds:
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)

        if eval_metric is not None:
            self.model.set_params(eval_metric=eval_metric)

        if eval_set:
            if early_stopping_rounds:
                print(
                    f"Training XGBoost with early stopping (patience={early_stopping_rounds})."
                )
            else:
                print("Training XGBoost with validation monitoring.")
        else:
            print("Training XGBoost regressor...")

        fit_kwargs.update(extra_fit_kwargs)
        self.model.fit(X_train, y_train, **fit_kwargs)

    def predict(self, X: Any) -> Any:
        """Generate predictions."""
        return self.model.predict(X)

    def save(self, path: str | Path) -> Path:
        """Persist the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        print(f"Saved XGBoost model to {path}.")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "XGBModel":
        """Load a model from disk."""
        path = Path(path)
        instance = cls()
        instance.model.load_model(path)
        print(f"Loaded XGBoost model from {path}.")
        return instance
