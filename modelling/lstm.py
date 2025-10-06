"""Baseline LSTM trainer for the dataset_dl feature table.

The script converts the engineered deep-learning dataset into fixed-length
sequences, trains a compact LSTM regressor, and writes predictions/metrics to
models/lstm/<run>/ (via output_utils.prepare_run_directory).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - helper guidance
    raise SystemExit("PyTorch is required for this script. Install with pip install torch." ) from exc

try:
    from .output_utils import prepare_run_directory
except ImportError:  # pragma: no cover - running as a script
    from output_utils import prepare_run_directory


DEFAULT_DATA_PATH = Path("model_datasets/dataset_dl.csv")
DEFAULT_TIMESTAMP = "timestamp"
DEFAULT_TARGET = "delivered_value"
DEFAULT_METER_COL = "meter_ui"
DEFAULT_OUTPUT_DIR = Path("models/lstm")
NON_FEATURE_COLUMNS = {"meter_ui", "nmi_ui", "timestamp", "meter_idx"}


def load_dataset(path: Path, timestamp_col: str, target_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' missing from dataset")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from dataset")
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col, target_col])
    df = df.sort_values(timestamp_col)
    return df


def detect_meter_column(df: pd.DataFrame, preferred: str) -> Optional[str]:
    if preferred in df.columns:
        return preferred
    for candidate in ("meter_ui", "nmi_ui", "meter_id"):
        if candidate in df.columns:
            return candidate
    return None


def aggregate_dataset(df: pd.DataFrame, timestamp_col: str, target_col: str, feature_cols: Sequence[str]) -> pd.DataFrame:
    agg_spec = {target_col: "sum"}
    for col in feature_cols:
        if col != target_col:
            agg_spec[col] = "mean"
    aggregated = df.groupby(timestamp_col, as_index=False).agg(agg_spec)
    return aggregated.sort_values(timestamp_col)


def build_sequences(values: np.ndarray, targets: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx in range(lookback, len(values)):
        X.append(values[idx - lookback : idx])
        y.append(targets[idx])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def create_datasets(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    feature_cols: Sequence[str],
    lookback: int,
    val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("val_ratio should be between 0 and 0.5 for temporal splits")
    values = df[feature_cols].astype(np.float32).values
    targets = df[target_col].astype(np.float32).values
    timestamps = df[timestamp_col].values

    split_row = int(len(df) * (1 - val_ratio))
    if split_row <= lookback or len(df) - split_row <= lookback:
        raise ValueError("Increase data size or adjust lookback/val_ratio; not enough samples after split")

    scaler = StandardScaler()
    scaler.fit(values[:split_row])
    values_scaled = scaler.transform(values)

    X, y = build_sequences(values_scaled, targets, lookback)
    target_times = timestamps[lookback:]

    split_seq = split_row - lookback
    X_train, y_train = X[:split_seq], y[:split_seq]
    X_val, y_val = X[split_seq:], y[split_seq:]
    val_times = target_times[split_seq:]

    if len(X_val) == 0:
        raise ValueError("Validation set is empty; adjust val_ratio or lookback")

    return X_train, y_train, X_val, y_val, val_times


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(-1)

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
    except TypeError:  # compatibility with older sklearn
        rmse = mean_squared_error(y_true, y_pred) ** 0.5

    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100.0
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small LSTM on dataset_dl.csv")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to dataset_dl.csv")
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP, help="Timestamp column name")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column name")
    parser.add_argument("--meter-column", default=DEFAULT_METER_COL, help="Meter identifier column")
    parser.add_argument("--meter", help="Meter ID to train (omit for first meter)")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate across meters before training")
    parser.add_argument("--lookback", type=int, default=24, help="Sequence length (hours)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of samples reserved for validation")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden units")
    parser.add_argument("--layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout between LSTM layers")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay for Adam optimizer")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (epochs)")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum validation loss improvement to reset patience")
    parser.add_argument("--clip-grad-norm", type=float, default=0.0, help="Gradient clipping norm (0 disables)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Computation device")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Base directory for outputs")
    parser.add_argument("--run-name", help="Optional run name for the output folder")
    parser.add_argument("--no-timestamp", action="store_true", help="Disable timestamp suffix in run directory name")
    return parser.parse_args()


def select_device(option: str) -> torch.device:
    if option == "cpu":
        return torch.device("cpu")
    if option == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    df = load_dataset(args.data_path, args.timestamp, args.target)
    meter_col = detect_meter_column(df, args.meter_column)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != args.target]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding identifiers and target")

    if args.aggregate:
        data = aggregate_dataset(df, args.timestamp, args.target, feature_cols)
        mode_label = "aggregate"
    else:
        if meter_col is None:
            raise ValueError("Meter column not detected; rerun with --aggregate or provide the correct column name")
        data = df[df[meter_col].notna()].copy()
        if args.meter is None:
            if data.empty:
                raise ValueError("Dataset has no valid meter rows")
            meter_value = str(data[meter_col].iloc[0])
            print(f"No --meter provided; defaulting to meter '{meter_value}'")
        else:
            meter_value = args.meter
        data = data[data[meter_col] == meter_value].copy()
        if data.empty:
            raise ValueError(f"Meter '{meter_value}' has no rows in dataset")
        data = data.sort_values(args.timestamp)
        drop_cols = [col for col in (meter_col, "nmi_ui", "meter_idx") if col in data.columns]
        data = data.drop(columns=drop_cols)
        mode_label = str(meter_value)

    data = data.dropna(subset=feature_cols + [args.target])
    data = data.sort_values(args.timestamp)

    X_train, y_train, X_val, y_val, val_times = create_datasets(
        data,
        args.timestamp,
        args.target,
        feature_cols,
        lookback=args.lookback,
        val_ratio=args.val_ratio,
    )

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_dataset = SequenceDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = LSTMRegressor(X_train.shape[-1], args.hidden_size, args.layers, args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
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
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                val_losses.append(criterion(preds, y_batch).item())

        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
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
    preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            batch_preds = model(X_batch).cpu().numpy().squeeze(-1)
            preds.append(batch_preds)
    y_pred = np.concatenate(preds)
    y_true = y_val

    metrics = evaluate_predictions(y_true, y_pred)
    print("Validation metrics:", metrics)

    epochs_trained = len(history)
    best_epoch = int(min(history, key=lambda row: row['val_loss'])['epoch']) if history else None

    run_dir = prepare_run_directory(args.output_dir, args.run_name, timestamp=not args.no_timestamp)
    print(f"Saving artifacts to {run_dir}")

    metrics_path = run_dir / f"metrics_{mode_label}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({
            "mode": mode_label,
            "metrics": metrics,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "lookback": args.lookback,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "min_delta": args.min_delta,
        }, f, indent=2)

    history_path = run_dir / f"training_history_{mode_label}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    preds_path = run_dir / f"predictions_{mode_label}.csv"
    pd.DataFrame({
        args.timestamp: val_times,
        "y_true": y_true,
        "y_pred": y_pred,
    }).to_csv(preds_path, index=False)

    model_path = run_dir / f"lstm_{mode_label}.pt"
    torch.save(model.state_dict(), model_path)

    print(f"Artifacts saved: {metrics_path.name}, {preds_path.name}, {model_path.name}, {history_path.name}")


if __name__ == "__main__":
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    main()
