"""Aggregated LSTM model implementation for sequence forecasting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from core.data.preperation import load_aggregated_series, load_raw_series, split_by_time_markers
from src.core.evaluation.metrics import metric_dict
from src.core.models import ModelBase, ModelInfo, TrainResult
from src.core.models.utils import assign_frequency, inverse_log
from src.core.registry import register


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.x = torch.from_numpy(sequences).unsqueeze(-1).float()
        self.y = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


def _create_sequences(series: pd.Series, window: int, mean: float, std: float) -> Tuple[np.ndarray, np.ndarray, list]:
    if len(series) <= window:
        return np.empty((0, window), dtype=np.float32), np.empty((0,), dtype=np.float32), []

    values = series.values.astype(np.float32)
    if std == 0:
        std = 1.0
    normalized = (values - mean) / std
    sequences = []
    targets = []
    indices = []
    for i in range(window, len(normalized)):
        sequences.append(normalized[i - window : i])
        targets.append(normalized[i])
        indices.append(series.index[i])
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32), indices


def _create_sequences_with_context(
    series: pd.Series,
    context: pd.Series,
    window: int,
    mean: float,
    std: float,
) -> Tuple[np.ndarray, np.ndarray, list]:
    if series.empty:
        return np.empty((0, window), dtype=np.float32), np.empty((0,), dtype=np.float32), []

    combined = pd.concat([context.tail(window), series])
    sequences, targets, indices = _create_sequences(combined, window, mean, std)
    series_index = set(series.index)
    filtered_sequences = []
    filtered_targets = []
    filtered_indices = []
    for seq, tgt, idx in zip(sequences, targets, indices):
        if idx in series_index:
            filtered_sequences.append(seq)
            filtered_targets.append(tgt)
            filtered_indices.append(idx)

    if not filtered_sequences:
        return np.empty((0, window), dtype=np.float32), np.empty((0,), dtype=np.float32), []

    return (
        np.asarray(filtered_sequences, dtype=np.float32),
        np.asarray(filtered_targets, dtype=np.float32),
        filtered_indices,
    )


def _denormalize(array: np.ndarray, mean: float, std: float) -> np.ndarray:
    return array * std + mean


@dataclass
class PreparedSeries:
    dataset_mode: str
    id_col: Optional[str]
    train_map: Dict[str | None, pd.Series]
    val_map: Dict[str | None, pd.Series]
    test_map: Dict[str | None, pd.Series]
    train_series: pd.Series
    val_series: pd.Series
    test_series: pd.Series


@register
class LSTMModel(ModelBase):
    info = ModelInfo(
        name="lstm",
        display_name="LSTM Sequence Model",
        default_train_config=Path("configs/model/lstm.yaml"),
        default_tune_config=Path("configs/model/lstm.yaml"),
        description="PyTorch LSTM trained on aggregated energy usage sequences.",
        tags=("deep-learning", "pytorch"),
    )

    def _prepare_series(self, data: pd.DataFrame) -> PreparedSeries:
        target_col = self.dataset_config.get("target_col", "DELIVERED_VALUE")
        dataset_mode = str(self.config.get("dataset_mode", "aggregated")).lower()
        ts_col = "timestamp"

        if dataset_mode == "raw":
            raw_df = load_raw_series(self.dataset_config, data)
            id_col = self.dataset_config.get("id_col")
            if not id_col:
                raise ValueError("dataset_mode='raw' requires 'id_col' in dataset configuration.")
            frames = split_by_time_markers(raw_df, self.dataset_config, ts_col=ts_col)
            train_map = self._build_series_map(frames.train, target_col, id_col)
            val_map = self._build_series_map(frames.val, target_col, id_col)
            test_map = self._build_series_map(frames.test, target_col, id_col)
            self._assign_frequency_to_map(train_map)
            self._assign_frequency_to_map(val_map)
            self._assign_frequency_to_map(test_map)
            train_series = self._combine_series_map(train_map, dataset_mode, id_col, target_col)
            val_series = self._combine_series_map(val_map, dataset_mode, id_col, target_col)
            test_series = self._combine_series_map(test_map, dataset_mode, id_col, target_col)
            return PreparedSeries(
                dataset_mode=dataset_mode,
                id_col=id_col,
                train_map=train_map,
                val_map=val_map,
                test_map=test_map,
                train_series=train_series,
                val_series=val_series,
                test_series=test_series,
            )

        aggregated = load_aggregated_series(self.dataset_config, data)
        frames = split_by_time_markers(aggregated, self.dataset_config, ts_col=ts_col)

        def _series(frame: pd.DataFrame) -> pd.Series:
            if frame is None or frame.empty:
                return pd.Series(dtype=float)
            series = frame.set_index(ts_col)[target_col].astype(float)
            series.index = pd.to_datetime(series.index)
            series.index.name = ts_col
            series.name = target_col
            return series

        train_series = _series(frames.train)
        val_series = _series(frames.val)
        test_series = _series(frames.test)
        assign_frequency(
            [train_series.to_frame(), val_series.to_frame(), test_series.to_frame()],
            self.dataset_config.get("freq"),
        )
        train_map = {"__all__": train_series}
        val_map = {"__all__": val_series}
        test_map = {"__all__": test_series}
        return PreparedSeries(
            dataset_mode=dataset_mode,
            id_col=None,
            train_map=train_map,
            val_map=val_map,
            test_map=test_map,
            train_series=train_series,
            val_series=val_series,
            test_series=test_series,
        )

    def _build_series_map(
        self,
        frame: pd.DataFrame,
        target_col: str,
        id_col: str,
    ) -> Dict[str, pd.Series]:
        if frame is None or frame.empty:
            return {}
        subset = frame[[id_col, "timestamp", target_col]].copy()
        subset["timestamp"] = pd.to_datetime(subset["timestamp"], errors="coerce")
        subset = subset.dropna(subset=["timestamp", target_col])
        series_map: Dict[str, pd.Series] = {}
        for meter_id, group in subset.groupby(id_col):
            series = (
                group.sort_values("timestamp")
                .set_index("timestamp")[target_col]
                .astype(float)
            )
            series.index.name = "timestamp"
            series.name = target_col
            series_map[str(meter_id)] = series
        return series_map

    def _assign_frequency_to_map(self, series_map: Dict[str | None, pd.Series]) -> None:
        frames = [
            series.to_frame(name=series.name or "value")
            for series in series_map.values()
            if not series.empty
        ]
        if frames:
            try:
                assign_frequency(frames, self.dataset_config.get("freq"))
            except ValueError:
                # Some meters may have irregular timestamps; skip enforcing freq.
                pass

    @staticmethod
    def _combine_series_map(
        series_map: Dict[str | None, pd.Series],
        dataset_mode: str,
        id_col: Optional[str],
        target_col: str,
    ) -> pd.Series:
        if not series_map:
            return pd.Series(dtype=float)
        if dataset_mode == "raw":
            meters: list[str] = []
            timestamps: list[pd.Timestamp] = []
            values: list[float] = []
            for meter_id, series in series_map.items():
                if series.empty:
                    continue
                meters.extend([meter_id] * len(series))
                timestamps.extend(pd.to_datetime(series.index))
                values.extend(series.values)
            if not values:
                return pd.Series(dtype=float)
            idx = pd.MultiIndex.from_arrays(
                [meters, timestamps],
                names=[id_col or "meter_id", "timestamp"],
            )
            combined = pd.Series(values, index=idx, name=target_col)
            return combined.sort_index()
        series = next(iter(series_map.values()))
        return series.sort_index()

    @staticmethod
    def _merge_series_maps(series_maps: Sequence[Dict[str | None, pd.Series]]) -> Dict[str | None, pd.Series]:
        merged: Dict[str | None, pd.Series] = {}
        for series_map in series_maps:
            for meter_id, series in series_map.items():
                if meter_id not in merged:
                    merged[meter_id] = series
                else:
                    merged[meter_id] = pd.concat([merged[meter_id], series]).sort_index()
        return merged

    def _format_index_entries(
        self,
        meter_id: str | None,
        timestamps: Sequence[pd.Timestamp],
        dataset_mode: str,
    ) -> list[tuple[str | None, pd.Timestamp]]:
        meter_label = str(meter_id) if (dataset_mode == "raw" and meter_id is not None) else None
        return [(meter_label, pd.to_datetime(ts)) for ts in timestamps]

    def _build_prediction_sequences(
        self,
        series_map: Dict[str | None, pd.Series],
        window: int,
        mean: float,
        std: float,
        dataset_mode: str,
    ) -> tuple[np.ndarray, list[tuple[str | None, pd.Timestamp]]]:
        seq_parts: list[np.ndarray] = []
        idx_entries: list[tuple[str | None, pd.Timestamp]] = []
        for meter_id, series in series_map.items():
            seq, _, idx = _create_sequences(series, window, mean, std)
            if len(seq) == 0:
                continue
            seq_parts.append(seq)
            idx_entries.extend(self._format_index_entries(meter_id, idx, dataset_mode))
        sequences = np.concatenate(seq_parts, axis=0) if seq_parts else np.empty((0, window), dtype=np.float32)
        return sequences, idx_entries

    def _prepare_sequences(
        self,
        train_map: Dict[str | None, pd.Series],
        val_map: Dict[str | None, pd.Series],
        test_map: Dict[str | None, pd.Series],
        window: int,
        dataset_mode: str,
    ) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, list[tuple[str | None, pd.Timestamp]]]], float, float]:
        def _concat_series_map(series_map: Dict[str | None, pd.Series]) -> pd.Series:
            series_list = [series for series in series_map.values() if not series.empty]
            if not series_list:
                return pd.Series(dtype=float)
            return pd.concat(series_list).sort_index()

        train_concat = _concat_series_map(train_map)
        mean = float(train_concat.mean()) if not train_concat.empty else 0.0
        std = float(train_concat.std()) if not train_concat.empty else 1.0
        if std == 0 or np.isnan(std):
            std = 1.0

        empty_series = pd.Series(dtype=float)

        def _stack_results(
            entries: list[tuple[np.ndarray, np.ndarray, list[tuple[str | None, pd.Timestamp]]]]
        ) -> Tuple[np.ndarray, np.ndarray, list[tuple[str | None, pd.Timestamp]]]:
            if not entries:
                return (
                    np.empty((0, window), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    [],
                )
            seqs = [seq for seq, _, _ in entries if len(seq) > 0]
            tgts = [tgt for _, tgt, _ in entries if len(tgt) > 0]
            idx = [idx for _, _, idx in entries if idx]
            sequences = np.concatenate(seqs, axis=0) if seqs else np.empty((0, window), dtype=np.float32)
            targets = np.concatenate(tgts, axis=0) if tgts else np.empty((0,), dtype=np.float32)
            indices: list[tuple[str | None, pd.Timestamp]] = []
            for chunk in idx:
                indices.extend(chunk)
            return sequences, targets, indices

        train_entries = []
        for meter_id, series in train_map.items():
            seq, tgt, idx = _create_sequences(series, window, mean, std)
            if len(seq) == 0:
                continue
            train_entries.append((seq, tgt, self._format_index_entries(meter_id, idx, dataset_mode)))

        val_entries = []
        for meter_id, series in val_map.items():
            context = train_map.get(meter_id, empty_series)
            seq, tgt, idx = _create_sequences_with_context(series, context, window, mean, std)
            if len(seq) == 0:
                continue
            val_entries.append((seq, tgt, self._format_index_entries(meter_id, idx, dataset_mode)))

        test_entries = []
        for meter_id, series in test_map.items():
            context_parts = []
            if meter_id in train_map:
                context_parts.append(train_map[meter_id])
            if meter_id in val_map:
                context_parts.append(val_map[meter_id])
            context_series = pd.concat(context_parts) if context_parts else empty_series
            seq, tgt, idx = _create_sequences_with_context(series, context_series, window, mean, std)
            if len(seq) == 0:
                continue
            test_entries.append((seq, tgt, self._format_index_entries(meter_id, idx, dataset_mode)))

        sequence_map = {
            "train": _stack_results(train_entries),
            "val": _stack_results(val_entries),
            "test": _stack_results(test_entries),
        }
        return sequence_map, mean, std

    def _select_device(self) -> torch.device:
        """Return the compute device honoring config override (cpu/cuda/auto)."""
        preference = str(self.config.get("device", "auto")).lower()
        if preference in ("auto", "default", ""):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if preference in ("cuda", "gpu"):
            if not torch.cuda.is_available():
                raise RuntimeError("LSTM config requests 'cuda' but no CUDA-capable device is available.")
            return torch.device("cuda")
        if preference == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unsupported device preference '{preference}'. Use 'auto', 'cpu', or 'cuda'.")

    def _build_model(self, hidden_size: int, num_layers: int, dropout: float) -> nn.Module:
        return LSTMRegressor(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    def _train_network(
        self,
        sequence_map: Dict[str, Tuple[np.ndarray, np.ndarray, list]],
        hidden_size: int,
        num_layers: int,
        dropout: float,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        device: torch.device,
    ) -> nn.Module:
        train_sequences, train_targets, _ = sequence_map["train"]
        if len(train_sequences) == 0:
            raise ValueError("Training data does not contain enough history for the specified window size.")

        model = self._build_model(hidden_size, num_layers, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        train_loader = DataLoader(SequenceDataset(train_sequences, train_targets), batch_size=batch_size, shuffle=True)

        val_sequences, val_targets, _ = sequence_map["val"]
        val_dataset = SequenceDataset(val_sequences, val_targets) if len(val_sequences) > 0 else None
        val_loader: Optional[DataLoader] = None
        if val_dataset is not None:
            # Keep validation on GPU-friendly mini-batches instead of one huge tensor.
            val_batch_size = max(1, min(batch_size, len(val_dataset)))
            val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        best_state = None
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

            if val_loader is None:
                best_state = model.state_dict()
                continue

            model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                val_count = 0
                for val_inputs, val_targets_tensor in val_loader:
                    val_inputs = val_inputs.to(device)
                    val_targets_tensor = val_targets_tensor.to(device)
                    val_preds = model(val_inputs)
                    batch_loss = criterion(val_preds, val_targets_tensor).item()
                    batch_size_actual = val_inputs.size(0)
                    val_loss_sum += batch_loss * batch_size_actual
                    val_count += batch_size_actual
                val_loss = val_loss_sum / max(1, val_count)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model

    def _produce_predictions(
        self,
        model: nn.Module,
        sequence_map: Dict[str, Tuple[np.ndarray, np.ndarray, list[tuple[str | None, pd.Timestamp]]]],
        mean: float,
        std: float,
        device: torch.device,
        dataset_mode: str,
        id_col: Optional[str],
    ) -> Dict[str, pd.Series]:
        model.eval()
        outputs: Dict[str, pd.Series] = {}
        for split, (sequences, _, indices) in sequence_map.items():
            if len(sequences) == 0:
                outputs[split] = pd.Series(dtype=float)
                continue
            preds = self._batched_forward(model, sequences, device)
            preds = _denormalize(preds, mean, std)
            preds = np.asarray(preds).reshape(-1)
            index = self._build_prediction_index(indices, dataset_mode, id_col)
            outputs[split] = pd.Series(preds, index=index)
        return outputs

    @staticmethod
    def _build_prediction_index(
        entries: list[tuple[str | None, pd.Timestamp]],
        dataset_mode: str,
        id_col: Optional[str],
    ) -> pd.Index | pd.MultiIndex:
        if not entries:
            if dataset_mode == "raw":
                return pd.MultiIndex.from_arrays(
                    [[], []],
                    names=[id_col or "meter_id", "timestamp"],
                )
            return pd.DatetimeIndex([], name="timestamp")
        meters, timestamps = zip(*entries)
        timestamps = pd.to_datetime(list(timestamps))
        if dataset_mode == "raw":
            meter_labels = [meter if meter is not None else "" for meter in meters]
            return pd.MultiIndex.from_arrays(
                [meter_labels, timestamps],
                names=[id_col or "meter_id", "timestamp"],
            )
        return pd.DatetimeIndex(timestamps, name="timestamp")

    def train(self, data: pd.DataFrame, output_dir: Path) -> TrainResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = output_dir / "artifacts"
        reports_dir = output_dir / "reports"
        predictions_dir = output_dir / "predictions"
        for path in (artifacts_dir, reports_dir, predictions_dir):
            path.mkdir(parents=True, exist_ok=True)

        device = self._select_device()
        quick_mode = bool(self.config.get("_runtime_quick"))

        window = int(self.config.get("window", 168))
        hidden_size = int(self.config.get("hidden_size", 64))
        num_layers = int(self.config.get("num_layers", 2))
        dropout = float(self.config.get("dropout", 0.1))
        epochs = int(self.config.get("epochs", 30))
        learning_rate = float(self.config.get("learning_rate", 1e-3))
        batch_size = int(self.config.get("batch_size", 64))

        if quick_mode:
            window = min(window, 72)
            hidden_size = min(hidden_size, 32)
            num_layers = min(num_layers, 1)
            epochs = min(epochs, 10)
            batch_size = min(batch_size, 32)

        series_data = self._prepare_series(data)
        sequences, mean, std = self._prepare_sequences(
            series_data.train_map,
            series_data.val_map,
            series_data.test_map,
            window,
            series_data.dataset_mode,
        )

        model = self._train_network(
            sequences,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device,
        )

        preds = self._produce_predictions(
            model,
            sequences,
            mean,
            std,
            device,
            series_data.dataset_mode,
            series_data.id_col,
        )

        metrics = {}
        if not preds["train"].empty:
            metrics["Train"] = metric_dict(series_data.train_series.loc[preds["train"].index], preds["train"])
        if not preds["val"].empty:
            metrics["Val"] = metric_dict(series_data.val_series.loc[preds["val"].index], preds["val"])
        if not preds["test"].empty:
            metrics["Test"] = metric_dict(series_data.test_series.loc[preds["test"].index], preds["test"])

        metrics_path = reports_dir / "metrics.csv"
        pd.DataFrame.from_records(
            [{"split": split, **values} for split, values in metrics.items()]
        ).to_csv(metrics_path, index=False)

        preds["train"].to_frame("y_hat").to_csv(predictions_dir / "train.csv")
        if not preds["val"].empty:
            preds["val"].to_frame("y_hat").to_csv(predictions_dir / "val.csv")
        if not preds["test"].empty:
            preds["test"].to_frame("y_hat").to_csv(predictions_dir / "test.csv")

        model_path = artifacts_dir / "lstm.pt"
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "window": window}, model_path)

        config_path = artifacts_dir / "lstm_config.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "window": window,
                },
                handle,
                indent=2,
            )

        primary_metrics = metrics.get("Test", metrics.get("Val", {}))
        return TrainResult(
            fitted_model=None,
            metrics=primary_metrics,
            artifacts={
                "metrics": metrics_path,
                "predictions_dir": predictions_dir,
            },
            model_path=model_path,
        )

    def tune(self, data: pd.DataFrame, output_dir: Path) -> dict[str, float]:
        output_dir.mkdir(parents=True, exist_ok=True)
        tuning_dir = output_dir / "tuning"
        tuning_dir.mkdir(parents=True, exist_ok=True)

        series_data = self._prepare_series(data)
        if series_data.val_series.empty:
            raise RuntimeError("Validation data is required for LSTM tuning; adjust dataset splits.")

        quick_mode = bool(self.config.get("_runtime_quick"))
        base_window = int(self.config.get("window", 168))
        base_hidden = int(self.config.get("hidden_size", 64))
        base_layers = int(self.config.get("num_layers", 2))

        candidate_grid = self.config.get("tune_grid", {})
        if not candidate_grid:
            candidate_grid = {
                "window": [base_window],
                "hidden_size": [base_hidden, max(16, base_hidden // 2)],
                "learning_rate": [self.config.get("learning_rate", 1e-3)],
            }

        if quick_mode:
            candidate_grid = {key: values[:1] for key, values in candidate_grid.items()}

        device = self._select_device()
        results = []
        best_score = float("inf")
        best_params = None

        for window in candidate_grid.get("window", [base_window]):
            window = int(window)
            sequences, mean, std = self._prepare_sequences(
                series_data.train_map,
                series_data.val_map,
                series_data.test_map,
                window,
                series_data.dataset_mode,
            )
            if len(sequences["val"][0]) == 0:
                continue

            hidden_sizes = candidate_grid.get("hidden_size", [base_hidden])
            learning_rates = candidate_grid.get("learning_rate", [self.config.get("learning_rate", 1e-3)])

            for hidden_size in hidden_sizes:
                for lr in learning_rates:
                    try:
                        model = self._train_network(
                            sequences,
                            hidden_size=int(hidden_size),
                            num_layers=base_layers,
                            dropout=float(self.config.get("dropout", 0.1)),
                            epochs=min(10, int(self.config.get("epochs", 30))),
                            learning_rate=float(lr),
                            batch_size=min(32, int(self.config.get("batch_size", 64))),
                            device=device,
                        )
                        preds = self._produce_predictions(
                            model,
                            sequences,
                            mean,
                            std,
                            device,
                            series_data.dataset_mode,
                            series_data.id_col,
                        )
                        val_preds = preds["val"]
                        val_actual = series_data.val_series.loc[val_preds.index]
                        score = metric_dict(val_actual, val_preds)["WAPE"]
                    except Exception as exc:  # noqa: BLE001
                        print(f"[LSTM] Tuning candidate failed (window={window}, hidden={hidden_size}, lr={lr}): {exc}")
                        continue

                    record = {
                        "window": window,
                        "hidden_size": int(hidden_size),
                        "learning_rate": float(lr),
                        "WAPE": score,
                    }
                    results.append(record)
                    if score < best_score:
                        best_score = score
                        best_params = record

        trials_path = tuning_dir / "lstm_trials.csv"
        pd.DataFrame(results).to_csv(trials_path, index=False)

        best_path = tuning_dir / "lstm_best.json"
        if best_params:
            with best_path.open("w", encoding="utf-8") as handle:
                json.dump(best_params, handle, indent=2)
        else:
            best_path.write_text("[]", encoding="utf-8")

        return {"WAPE": best_score if best_params else float("inf")}

    def predict(
        self,
        model_path: Path,
        data: pd.DataFrame,
        horizon: int | None = None,
    ) -> pd.DataFrame:
        checkpoint = torch.load(model_path, map_location="cpu")
        config_path = model_path.with_name("lstm_config.json")
        if not config_path.exists():
            raise FileNotFoundError("Missing LSTM configuration file beside checkpoint.")
        config = json.loads(config_path.read_text(encoding="utf-8"))

        window = int(checkpoint.get("window", config.get("window", 168)))
        hidden_size = int(config.get("hidden_size", 64))
        num_layers = int(config.get("num_layers", 2))
        dropout = float(config.get("dropout", 0.1))
        mean = float(checkpoint.get("mean", 0.0))
        std = float(checkpoint.get("std", 1.0))

        series_data = self._prepare_series(data)
        merged_map = self._merge_series_maps(
            [series_data.train_map, series_data.val_map, series_data.test_map]
        )
        target_col = self.dataset_config.get("target_col", "DELIVERED_VALUE")
        combined_series = self._combine_series_map(
            merged_map,
            series_data.dataset_mode,
            series_data.id_col,
            target_col,
        )
        if combined_series.empty:
            if series_data.dataset_mode == "raw":
                meter_col = series_data.id_col or "meter_id"
                return pd.DataFrame(columns=[meter_col, "timestamp", "y_hat", "y"])
            return pd.DataFrame(columns=["timestamp", "y_hat", "y"])

        if series_data.dataset_mode == "raw":
            self._assign_frequency_to_map(merged_map)
        else:
            assign_frequency([combined_series.to_frame(name=target_col)], self.dataset_config.get("freq"))

        sequences, indices = self._build_prediction_sequences(
            merged_map,
            window,
            mean,
            std,
            series_data.dataset_mode,
        )
        if len(sequences) == 0:
            if series_data.dataset_mode == "raw":
                meter_col = series_data.id_col or "meter_id"
                return pd.DataFrame(columns=[meter_col, "timestamp", "y_hat", "y"])
            return pd.DataFrame(columns=["timestamp", "y_hat", "y"])

        model = self._build_model(hidden_size, num_layers, dropout)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        device = self._select_device()
        model.to(device)

        preds = self._batched_forward(model, sequences, device)
        preds = _denormalize(np.asarray(preds).reshape(-1), mean, std)

        index = self._build_prediction_index(indices, series_data.dataset_mode, series_data.id_col)
        pred_series = pd.Series(preds, index=index)
        actual_series = combined_series.loc[pred_series.index]

        if series_data.dataset_mode == "raw":
            df_out = pred_series.rename("y_hat").reset_index()
            df_out["y"] = actual_series.values
            return df_out

        return pd.DataFrame(
            {
                "timestamp": pred_series.index,
                "y_hat": pred_series.values,
                "y": actual_series.values,
            }
        )

    def _batched_forward(
        self,
        model: nn.Module,
        sequences: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        if len(sequences) == 0:
            return np.empty((0,), dtype=np.float32)

        default_eval_batch = int(self.config.get("eval_batch_size", self.config.get("batch_size", 256)))
        batch_size = max(1, min(int(default_eval_batch), len(sequences)))
        preds: list[np.ndarray] = []

        for start in range(0, len(sequences), batch_size):
            end = start + batch_size
            batch = torch.from_numpy(sequences[start:end]).unsqueeze(-1).float().to(device)
            with torch.no_grad():
                batch_preds = model(batch).cpu().numpy()
            preds.append(batch_preds)

        return np.concatenate(preds, axis=0)
