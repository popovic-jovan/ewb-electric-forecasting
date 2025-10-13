"""LSTM trainer for the pre-built dataset_lstm.npz sequences.

The feature engineering pipeline now produces ready-to-train sliding window
sequences along with per-meter scaling statistics. This script loads those
artefacts, performs a temporal train/validation split, and fits a compact LSTM
regressor whose predictions and metrics are written to models/lstm/<run>/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for this script. Install with pip install torch.") from exc

try:
    from .output_utils import prepare_run_directory
except ImportError:  # pragma: no cover
    from output_utils import prepare_run_directory


DEFAULT_DATA_PATH = Path("model_datasets/dataset_lstm.npz")
DEFAULT_SCALER_PATH = Path("model_datasets/lstm_scalers.json")
DEFAULT_OUTPUT_DIR = Path("models/lstm")


@dataclass
class SequenceBundle:
    X: np.ndarray
    y: np.ndarray
    meter: np.ndarray
    timestamps: np.ndarray
    feature_order: List[str]
    lookback: int


@dataclass
class SequenceSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    val_timestamps: np.ndarray
    val_meter: np.ndarray


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(-1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.output(last)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:  # pragma: no cover - compatibility
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100.0
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


def load_sequence_bundle(path: Path) -> SequenceBundle:
    if not path.exists():
        raise FileNotFoundError(f"Sequence dataset not found: {path}")
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    meter = data["meter"].astype(str)
    timestamps = pd.to_datetime(data["timestamps"]).to_numpy()
    feature_order = data["feature_order"].tolist()
    lookback = int(data["lookback"][0]) if "lookback" in data else X.shape[1]
    return SequenceBundle(X=X, y=y, meter=meter, timestamps=timestamps, feature_order=feature_order, lookback=lookback)


def filter_bundle(bundle: SequenceBundle, meter: Optional[str]) -> SequenceBundle:
    if meter is None or meter.upper() == "ALL":
        return bundle
    mask = bundle.meter == meter
    if not mask.any():
        raise ValueError(f"No sequences found for meter '{meter}'.")
    return SequenceBundle(
        X=bundle.X[mask],
        y=bundle.y[mask],
        meter=bundle.meter[mask],
        timestamps=bundle.timestamps[mask],
        feature_order=bundle.feature_order,
        lookback=bundle.lookback,
    )


def order_bundle(bundle: SequenceBundle) -> SequenceBundle:
    order = np.argsort(bundle.timestamps.astype("datetime64[ns]"))
    return SequenceBundle(
        X=bundle.X[order],
        y=bundle.y[order],
        meter=bundle.meter[order],
        timestamps=bundle.timestamps[order],
        feature_order=bundle.feature_order,
        lookback=bundle.lookback,
    )


def temporal_split(bundle: SequenceBundle, val_ratio: float) -> SequenceSplit:
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("val_ratio must be between 0 and 0.5 for temporal splits.")
    n_samples = bundle.X.shape[0]
    split_idx = int(np.floor(n_samples * (1.0 - val_ratio)))
    if split_idx <= 0 or split_idx >= n_samples:
        raise ValueError("Not enough samples to perform the requested validation split.")
    return SequenceSplit(
        X_train=bundle.X[:split_idx],
        y_train=bundle.y[:split_idx],
        X_val=bundle.X[split_idx:],
        y_val=bundle.y[split_idx:],
        val_timestamps=bundle.timestamps[split_idx:],
        val_meter=bundle.meter[split_idx:],
    )


def select_device(option: str) -> torch.device:
    if option == "cpu":
        return torch.device("cpu")
    if option == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM on pre-built electricity usage sequences.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to dataset_lstm.npz")
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER_PATH, help="Optional scaler metadata JSON")
    parser.add_argument("--meter", help="Meter ID to filter (default: all meters)")
    parser.add_argument("--lookback", type=int, help="Expected sequence length; verified against dataset metadata.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of samples reserved for validation.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Number of LSTM hidden units.")
    parser.add_argument("--layers", type=int, default=1, help="Number of stacked LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout between LSTM layers.")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Adam optimiser learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay.")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (epochs).")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum validation loss improvement.")
    parser.add_argument("--clip-grad-norm", type=float, default=0.0, help="Gradient clipping norm (0 disables).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Computation device.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Base output directory.")
    parser.add_argument("--run-name", help="Optional run name to append to the output directory.")
    parser.add_argument("--no-timestamp", action="store_true", help="Disable timestamp suffix in output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    bundle = load_sequence_bundle(args.data_path)
    if args.lookback is not None and int(args.lookback) != bundle.lookback:
        raise ValueError(f"Dataset lookback ({bundle.lookback}) does not match --lookback ({args.lookback}).")

    bundle = filter_bundle(bundle, args.meter)
    bundle = order_bundle(bundle)

    split = temporal_split(bundle, args.val_ratio)

    if split.X_train.shape[0] == 0:
        raise ValueError("Training split is empty; reduce --val-ratio.")
    if split.X_val.shape[0] == 0:
        raise ValueError("Validation split is empty; increase --val-ratio.")

    train_loader = DataLoader(SequenceDataset(split.X_train, split.y_train), batch_size=args.batch_size, shuffle=True)
    val_dataset = SequenceDataset(split.X_val, split.y_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = LSTMRegressor(split.X_train.shape[-1], args.hidden_size, args.layers, args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: List[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: List[float] = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                val_losses.append(criterion(preds, y_batch).item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss + args.min_delta < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered at epoch {epoch:03d} (patience {args.patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            batch_preds = model(X_batch).cpu().numpy().squeeze(-1)
            preds.append(batch_preds)
    y_pred = np.concatenate(preds)
    y_true = split.y_val

    metrics = evaluate_predictions(y_true, y_pred)
    print("Validation metrics:", metrics)

    epochs_trained = len(history)
    best_epoch = int(min(history, key=lambda row: row["val_loss"])["epoch"]) if history else None

    mode_label = args.meter if args.meter else "all"
    run_dir = prepare_run_directory(args.output_dir, args.run_name, timestamp=not args.no_timestamp)
    print(f"Saving artifacts to {run_dir}")

    scaler_info: Optional[str] = None
    if args.scaler_path and args.scaler_path.exists():
        scaler_info = str(args.scaler_path.resolve())

    metrics_payload = {
        "mode": mode_label,
        "metrics": metrics,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "lookback": bundle.lookback,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "feature_order": bundle.feature_order,
        "num_train_sequences": int(split.X_train.shape[0]),
        "num_val_sequences": int(split.X_val.shape[0]),
    }
    if scaler_info:
        metrics_payload["scaler_path"] = scaler_info

    metrics_path = run_dir / f"metrics_{mode_label}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    history_path = run_dir / f"training_history_{mode_label}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    predictions_df = pd.DataFrame({
        "timestamp": pd.to_datetime(split.val_timestamps),
        "meter": split.val_meter,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    preds_path = run_dir / f"predictions_{mode_label}.csv"
    predictions_df.to_csv(preds_path, index=False)

    model_path = run_dir / f"lstm_{mode_label}.pt"
    torch.save(model.state_dict(), model_path)

    print(f"Artifacts saved: {metrics_path.name}, {preds_path.name}, {model_path.name}, {history_path.name}")


if __name__ == "__main__":
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    main()
