"""Aggregated XGBoost model implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna import TrialPruned
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from xgboost import XGBRegressor, core as xgb_core

from core.data.preperation import load_aggregated_series, load_raw_series, split_by_time_markers
from src.core.models import ModelBase, ModelInfo, TrainResult, XGBModel
from src.core.models.utils import prepare_feature_splits
from src.core.registry import register
from src.core.evaluation.metrics import metric_dict


@register
class XGBoostAggregatedModel(ModelBase):
    info = ModelInfo(
        name="xgboost",
        display_name="XGBoost Regressor",
        default_train_config=Path("configs/model/xgboost.yaml"),
        default_tune_config=Path("configs/model/xgboost.yaml"),
        description="Tree-based gradient boosted model trained on aggregated features.",
        tags=("gradient-boosting", "tabular"),
    )

    # ------------------------------------------------------------------ #
    # Dataset preparation helpers
    # ------------------------------------------------------------------ #
    def _prepare_datasets(self, data: pd.DataFrame):
        target_col = self.dataset_config.get("target_col", "DELIVERED_VALUE")
        ts_col = "timestamp"
        dataset_mode = str(self.config.get("dataset_mode", "aggregated")).lower()
        if dataset_mode == "raw":
            source_df = load_raw_series(self.dataset_config, data)
            group_col = self.dataset_config.get("id_col")
        else:
            source_df = load_aggregated_series(self.dataset_config, data)
            group_col = None

        frames = split_by_time_markers(source_df, self.dataset_config, ts_col=ts_col)
        feature_cfg = self.config.get("features", {})
        train_df, val_df, test_df, feature_cols = prepare_feature_splits(
            frames,
            feature_cfg,
            ts_col=ts_col,
            target_col=target_col,
            group_col=group_col,
        )
        return target_col, feature_cols, train_df, val_df, test_df, ts_col, group_col

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, data: pd.DataFrame, output_dir: Path) -> TrainResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = output_dir / "artifacts"
        reports_dir = output_dir / "reports"
        predictions_dir = output_dir / "predictions"
        for path in (artifacts_dir, reports_dir, predictions_dir):
            path.mkdir(parents=True, exist_ok=True)

        quick_mode = bool(self.config.get("_runtime_quick"))

        target_col, feature_cols, train_df, val_df, test_df, ts_col, group_col = self._prepare_datasets(data)

        if quick_mode:
            subset_hours = int(self.config.get("quick_history_hours", 24 * 7 * 4))
            if subset_hours > 0:
                train_df = train_df.tail(subset_hours)
            if not val_df.empty:
                val_df = val_df.tail(min(len(val_df), subset_hours // 4 or len(val_df)))
            if not test_df.empty:
                test_df = test_df.tail(min(len(test_df), subset_hours // 4 or len(test_df)))

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols] if not val_df.empty else None
        y_val = val_df[target_col] if not val_df.empty else None
        X_test = test_df[feature_cols] if not test_df.empty else None

        params = dict(self.config.get("params", {}))
        fit_params_cfg = dict(self.config.get("fit", {}))
        tuning_cfg = self.config.get("tuning", {})

        if "early_stopping_rounds" not in fit_params_cfg and tuning_cfg.get("early_stopping_rounds"):
            fit_params_cfg["early_stopping_rounds"] = tuning_cfg["early_stopping_rounds"]

        refit_on_full = bool(fit_params_cfg.pop("refit_on_full", True))
        verbose_flag = fit_params_cfg.pop("verbose", False)
        if verbose_flag and "verbosity" not in params:
            params["verbosity"] = 1
        if verbose_flag and "verbose" not in fit_params_cfg:
            fit_params_cfg["verbose"] = True
        fit_params = fit_params_cfg

        if quick_mode:
            params.setdefault("n_estimators", min(100, params.get("n_estimators", 500)))
            params["max_depth"] = min(params.get("max_depth", 6), 4)
            params.setdefault("learning_rate", 0.1)
            fit_params.setdefault("early_stopping_rounds", 10)

        model = XGBModel(params=params, fit_params=fit_params)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        best_iteration = getattr(model.model, "best_iteration", None)
        n_estimators_final = params.get("n_estimators")
        if best_iteration is not None:
            n_estimators_final = best_iteration + 1

        index_cols: list[str] = [ts_col]
        if group_col and group_col in train_df.columns:
            index_cols.insert(0, group_col)

        if refit_on_full and (not val_df.empty):
            combined = (
                pd.concat([train_df, val_df], ignore_index=True)
                .sort_values(index_cols)
                .reset_index(drop=True)
            )
        else:
            combined = train_df

        if refit_on_full and not combined.empty:
            params_refit = dict(params)
            if n_estimators_final is not None:
                params_refit["n_estimators"] = n_estimators_final
            refit_model = XGBModel(params=params_refit, fit_params={})
            refit_model.fit(
                combined[feature_cols],
                combined[target_col],
            )
            model = refit_model

        def _predict_frame(df_split: pd.DataFrame) -> pd.DataFrame:
            if df_split is None or df_split.empty:
                columns = index_cols + [target_col, "y_hat"]
                return pd.DataFrame(columns=columns)
            preds = model.predict(df_split[feature_cols])
            frame = df_split[index_cols + [target_col]].copy()
            frame["y_hat"] = preds
            return frame

        if refit_on_full and not val_df.empty:
            train_eval_df = (
                pd.concat([train_df, val_df], ignore_index=True)
                .sort_values(index_cols)
                .reset_index(drop=True)
            )
            train_label = "Train+Val"
        else:
            train_eval_df = train_df
            train_label = "Train"

        preds_train = _predict_frame(train_eval_df)
        preds_val = _predict_frame(val_df)
        preds_test = _predict_frame(test_df)

        metrics = {}
        if not preds_train.empty:
            metrics[train_label] = metric_dict(preds_train[target_col], preds_train["y_hat"])
        if not preds_val.empty and train_label != "Train+Val":
            metrics["Val"] = metric_dict(preds_val[target_col], preds_val["y_hat"])
        elif not preds_val.empty and train_label == "Train+Val":
            metrics["Val (pre-refit)"] = metric_dict(preds_val[target_col], preds_val["y_hat"])
        if not preds_test.empty:
            metrics["Test"] = metric_dict(preds_test[target_col], preds_test["y_hat"])

        metrics_path = reports_dir / "metrics.csv"
        pd.DataFrame.from_records(
            [{"split": split, **values} for split, values in metrics.items()]
        ).to_csv(metrics_path, index=False)

        preds_train.to_csv(predictions_dir / "train.csv", index=False)
        if not preds_val.empty:
            preds_val.to_csv(predictions_dir / "val.csv", index=False)
        if not preds_test.empty:
            preds_test.to_csv(predictions_dir / "test.csv", index=False)

        model_path = artifacts_dir / "model.json"
        model.save(model_path)

        feature_path = artifacts_dir / "feature_columns.json"
        with feature_path.open("w", encoding="utf-8") as handle:
            json.dump(feature_cols, handle, indent=2)

        primary_metrics = metrics.get("Test", metrics.get("Val", {}))
        return TrainResult(
            fitted_model=None,
            metrics=primary_metrics,
            artifacts={
                "metrics": metrics_path,
                "predictions_dir": predictions_dir,
                "feature_columns": feature_path,
            },
            model_path=model_path,
        )

    # ------------------------------------------------------------------ #
    # Hyperparameter tuning with Optuna
    # ------------------------------------------------------------------ #
    def tune(self, data: pd.DataFrame, output_dir: Path) -> dict[str, float]:
        output_dir.mkdir(parents=True, exist_ok=True)
        tuning_dir = output_dir / "tuning"
        tuning_dir.mkdir(parents=True, exist_ok=True)

        target_col, feature_cols, train_df, val_df, test_df, ts_col, group_col = self._prepare_datasets(data)
        index_cols: list[str] = [ts_col]
        if group_col and group_col in train_df.columns:
            index_cols.insert(0, group_col)

        train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        train_val_df = train_val_df.sort_values(index_cols).reset_index(drop=True)

        if train_val_df.empty:
            raise RuntimeError("Insufficient data for tuning; combined train/validation set is empty.")

        tuning_cfg = self.config.get("tuning", {})
        quick_mode = bool(self.config.get("_runtime_quick"))

        n_trials = int(
            tuning_cfg.get("n_trials", 50) if quick_mode else tuning_cfg.get("n_trials_final", tuning_cfg.get("n_trials", 150))
        )
        early_stop = int(tuning_cfg.get("early_stopping_rounds", 200))
        folds = int(tuning_cfg.get("cv_folds", 5))
        horizon = int(tuning_cfg.get("cv_horizon_hours", 168))
        gap = int(tuning_cfg.get("cv_gap_hours", 24))
        seed = int(tuning_cfg.get("seed", 42))
        n_jobs = int(tuning_cfg.get("n_jobs", 1))

        folds = self._generate_cv_folds(train_val_df, horizon, gap, folds)
        if not folds:
            raise RuntimeError("Unable to create CV folds with the provided horizon/gap settings.")

        sampler = TPESampler(seed=seed)
        pruner = MedianPruner(n_warmup_steps=2)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

        metrics_log: List[dict[str, float]] = []

        def objective(trial: optuna.trial.Trial) -> float:
            params = self._suggest_xgb_params(trial, seed, tuning_cfg)
            fold_scores = []
            fold_metrics = []

            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                X_train = train_val_df.loc[train_idx, feature_cols]
                y_train = train_val_df.loc[train_idx, target_col]
                X_valid = train_val_df.loc[val_idx, feature_cols]
                y_valid = train_val_df.loc[val_idx, target_col]

                model = self._create_regressor(params)
                try:
                    if early_stop:
                        model.set_params(early_stopping_rounds=early_stop)
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_valid, y_valid)],
                        verbose=False,
                    )
                except xgb_core.XGBoostError as exc:
                    if "GPU" in str(exc) and params.get("tree_method") == "gpu_hist":
                        params_fallback = dict(params)
                        params_fallback["tree_method"] = "hist"
                        params_fallback.pop("predictor", None)
                        model = self._create_regressor(params_fallback)
                        if early_stop:
                            model.set_params(early_stopping_rounds=early_stop)
                        model.fit(
                            X_train,
                            y_train,
                            eval_set=[(X_valid, y_valid)],
                            verbose=False,
                        )
                    else:
                        raise

                preds = model.predict(X_valid)
                metrics = metric_dict(y_valid, preds)
                fold_scores.append(metrics["WAPE"])
                fold_metrics.append(metrics)

                trial.report(metrics["WAPE"], step=fold_idx)
                if trial.should_prune():
                    raise TrialPruned()

            mean_metrics = {
                "WAPE": float(np.mean([m["WAPE"] for m in fold_metrics])),
                "MAE": float(np.mean([m["MAE"] for m in fold_metrics])),
                "RMSE": float(np.mean([m["RMSE"] for m in fold_metrics])),
            }
            trial.set_user_attr("metrics", mean_metrics)
            trial.set_user_attr("best_iteration", getattr(model, "best_iteration", None))
            return mean_metrics["WAPE"]

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)

        # Persist trial history
        trials_records = []
        for trial in study.trials:
            record = {
                "trial": trial.number,
                "value": trial.value,
                "state": trial.state.name,
            }
            record.update(trial.params)
            metrics_attr = trial.user_attrs.get("metrics")
            if metrics_attr:
                record.update({f"mean_{k.lower()}": v for k, v in metrics_attr.items()})
            trials_records.append(record)
        trials_df = pd.DataFrame(trials_records)
        (tuning_dir / "optuna_trials.csv").write_text(trials_df.to_csv(index=False), encoding="utf-8")

        best_params = study.best_trial.params
        best_metrics = study.best_trial.user_attrs.get("metrics", {})
        final_params = self._finalize_params(best_params, seed, tuning_cfg)

        holdout_metrics = {}
        if not test_df.empty:
            holdout_df = test_df.sort_values(index_cols).reset_index(drop=True)
            holdout_metrics = self._fit_and_evaluate_holdout(
                train_val_df,
                holdout_df,
                feature_cols,
                target_col,
                final_params,
                early_stop,
                horizon,
            )

        # Persist best params and holdout metrics
        best_path = tuning_dir / "xgboost_best.json"
        best_payload = {
            "params": final_params,
            "cv_metrics": best_metrics,
            "best_trial": study.best_trial.number,
            "holdout_metrics": holdout_metrics,
        }
        best_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

        result = {"cv_WAPE": float(study.best_value)}
        if holdout_metrics:
            result["holdout_WAPE"] = float(holdout_metrics.get("WAPE", float("nan")))
        return result

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(
        self,
        model_path: Path,
        data: pd.DataFrame,
        horizon: int | None = None,
    ) -> pd.DataFrame:
        if horizon:
            raise NotImplementedError("Future horizon prediction is not implemented for XGBoost.")
        model = XGBModel.load(model_path)

        feature_path = model_path.with_name("feature_columns.json")
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature metadata not found at {feature_path}. Re-run training to regenerate it."
            )
        feature_cols = json.loads(feature_path.read_text(encoding="utf-8"))

        target_col, feature_cols, train_df, val_df, test_df, ts_col, group_col = self._prepare_datasets(data)
        index_cols: list[str] = [ts_col]
        if group_col and group_col in train_df.columns:
            index_cols.insert(0, group_col)

        combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
        combined = combined.sort_values(index_cols).reset_index(drop=True)
        preds = model.predict(combined[feature_cols])

        df_out = combined[index_cols + [target_col]].copy()
        df_out["y_hat"] = preds
        return df_out

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _generate_cv_folds(
        self,
        df: pd.DataFrame,
        horizon: int,
        gap: int,
        folds: int,
    ) -> List[Tuple[Sequence[pd.Timestamp], Sequence[pd.Timestamp]]]:
        if folds <= 0:
            return []

        total_len = len(df)
        step = horizon + gap
        min_train = max(horizon, int(self.config.get("tuning", {}).get("min_cv_train_points", horizon * 2)))

        folds = min(folds, max(1, (total_len - min_train) // step))
        while folds > 1 and total_len - folds * step < min_train:
            folds -= 1

        initial_train_len = total_len - folds * step
        if initial_train_len < min_train:
            initial_train_len = min_train

        indices = df.index
        splits: List[Tuple[Sequence[pd.Timestamp], Sequence[pd.Timestamp]]] = []

        for fold_idx in range(folds):
            train_end = initial_train_len + fold_idx * step
            val_start = train_end + gap
            val_end = val_start + horizon
            if val_end > total_len:
                break
            train_idx = indices[:train_end]
            val_idx = indices[val_start:val_end]
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            splits.append((train_idx, val_idx))

        return splits

    def _suggest_xgb_params(
        self,
        trial: optuna.trial.Trial,
        seed: int,
        tuning_cfg: Mapping[str, object],
    ) -> dict[str, object]:
        use_gpu = bool(tuning_cfg.get("use_gpu", False))
        params: dict[str, object] = {
            "objective": "reg:squarederror",
            "tree_method": "gpu_hist" if use_gpu else "hist",
            "predictor": "gpu_predictor" if use_gpu else "auto",
            "n_estimators": int(tuning_cfg.get("n_estimators", 5000)),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "max_bin": trial.suggest_int("max_bin", 128, 512),
            "eval_metric": "rmse",
            "verbosity": 0,
            "random_state": seed,
        }

        grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if grow_policy == "lossguide":
            params["grow_policy"] = "lossguide"
            params["max_depth"] = 0
            params["max_leaves"] = trial.suggest_int("max_leaves", 32, 1024)
        else:
            params["grow_policy"] = "depthwise"
            params["max_depth"] = trial.suggest_int("max_depth", 3, 12)

        return params

    @staticmethod
    def _create_regressor(params: Mapping[str, object]) -> XGBRegressor:
        return XGBRegressor(**params)

    def _finalize_params(
        self,
        best_params: Mapping[str, object],
        seed: int,
        tuning_cfg: Mapping[str, object],
    ) -> dict[str, object]:
        final_params = dict(best_params)
        tree_method = final_params.get("tree_method", "hist")
        predictor = final_params.get("predictor", "auto")
        if tree_method == "gpu_hist":
            predictor = "gpu_predictor"
        final_params.update(
            {
                "objective": "reg:squarederror",
                "tree_method": tree_method,
                "predictor": predictor,
                "n_estimators": int(tuning_cfg.get("n_estimators", self.config.get("params", {}).get("n_estimators", 5000))),
                "verbosity": 0,
                "random_state": seed,
            }
        )
        return final_params

    def _fit_and_evaluate_holdout(
        self,
        train_val_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: str,
        params: Mapping[str, object],
        early_stop: int,
        horizon: int,
    ) -> dict[str, float]:
        if holdout_df.empty:
            return {}

        if len(train_val_df) > horizon:
            train_part = train_val_df.iloc[:-horizon]
            val_part = train_val_df.iloc[-horizon:]
            eval_set = [(val_part[feature_cols], val_part[target_col])]
        else:
            train_part = train_val_df
            eval_set = None

        model = self._create_regressor(params)
        try:
            if eval_set and early_stop:
                model.set_params(early_stopping_rounds=early_stop)
            model.fit(
                train_part[feature_cols],
                train_part[target_col],
                eval_set=eval_set,
                verbose=False,
            )
        except xgb_core.XGBoostError as exc:
            if "GPU" in str(exc) and params.get("tree_method") == "gpu_hist":
                params_fallback = dict(params)
                params_fallback["tree_method"] = "hist"
                params_fallback.pop("predictor", None)
                model = self._create_regressor(params_fallback)
                if eval_set and early_stop:
                    model.set_params(early_stopping_rounds=early_stop)
                model.fit(
                    train_part[feature_cols],
                    train_part[target_col],
                    eval_set=eval_set,
                    verbose=False,
                )
            else:
                raise

        holdout_preds = model.predict(holdout_df[feature_cols])
        return metric_dict(holdout_df[target_col], holdout_preds)
