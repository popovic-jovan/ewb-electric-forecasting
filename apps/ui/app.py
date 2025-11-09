"""Streamlit interface for browsing metrics and generating forecasts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.core.pipelines.predict import PredictArgs, run_predict  # noqa: E402
from src.core.registry import available_models  # noqa: E402
from src.core.io import load_yaml  # noqa: E402

DATASET_CONFIG_PATH = Path("configs/dataset.yaml")
MODELS_ROOT = Path("models")
DEFAULT_ARTIFACTS = {
    "xgboost": MODELS_ROOT / "xgboost" / "artifacts" / "model.json",
    "sarimax": MODELS_ROOT / "sarimax" / "artifacts" / "sarimax_model.pkl",
    "sarima": MODELS_ROOT / "sarima" / "artifacts" / "sarima_model.pkl",
    "prophet": MODELS_ROOT / "prophet" / "artifacts" / "prophet_model.json",
    "lstm": MODELS_ROOT / "lstm" / "artifacts" / "lstm.pt",
}

st.set_page_config(page_title="Electricity Forecast Demo", layout="wide")
st.title("Electricity Forecast Demo")

try:
    dataset_cfg = load_yaml(DATASET_CONFIG_PATH)
except FileNotFoundError as exc:
    st.error(f"Dataset configuration missing: {exc}")
    st.stop()

model_infos = list(available_models())
if not model_infos:
    st.warning("No models are registered yet. Train a model before using the dashboard.")
    st.stop()

model_names = [info.name for info in model_infos]
selected_model = st.sidebar.selectbox("Select model", model_names)
selected_info = next(info for info in model_infos if info.name == selected_model)

artifact_path = DEFAULT_ARTIFACTS.get(selected_model, MODELS_ROOT / selected_model / "artifacts")
if artifact_path.is_dir():
    # pick first file in artifacts directory
    files = sorted(artifact_path.glob("*"))
    artifact_path = files[0] if files else None
elif not artifact_path.exists():
    artifact_path = None

metrics_path = MODELS_ROOT / selected_model / "reports" / "metrics.csv"
predictions_dir = MODELS_ROOT / selected_model / "predictions"

st.subheader("Model Metadata")
st.write(f"**Display name:** {selected_info.display_name}")
if selected_info.description:
    st.write(selected_info.description)
st.write(f"**Default config:** {selected_info.default_train_config}")

st.subheader("Evaluation Metrics")
if metrics_path.exists():
    st.dataframe(pd.read_csv(metrics_path))
else:
    st.info("No metrics file found. Train the model to generate evaluation results.")

if predictions_dir.exists():
    st.subheader("Stored Predictions")
    available_files = sorted(predictions_dir.glob("*.csv"))
    if available_files:
        selected_pred_file = st.selectbox(
            "Choose a predictions file",
            available_files,
            format_func=lambda p: p.name,
        )
        stored_df = pd.read_csv(selected_pred_file)
        st.dataframe(stored_df.tail(20))
        if "timestamp" in stored_df.columns and "y_hat" in stored_df.columns:
            chart_df = stored_df.set_index("timestamp")[["y_hat"]]
            if "y" in stored_df.columns:
                chart_df["y"] = stored_df["y"]
            chart_df.index = pd.to_datetime(chart_df.index)
            st.line_chart(chart_df, height=300)
    else:
        st.info("No saved prediction files available yet.")

st.subheader("Upload Data for Live Forecasting")
uploaded_file = st.file_uploader(
    "Upload a CSV containing recent observations.",
    type=["csv"],
    help="Files should match the raw dataset schema (including timestamp and target columns).",
)

if uploaded_file is not None:
    if artifact_path is None:
        st.error("No trained checkpoint found for this model. Train the model first.")
        st.stop()

    data = pd.read_csv(uploaded_file)
    time_col = dataset_cfg.get("time_col")
    if time_col not in data.columns:
        st.error(f"Uploaded file must include the timestamp column '{time_col}'.")
        st.stop()

    data[time_col] = pd.to_datetime(data[time_col])

    try:
        predict_args = PredictArgs(
            model=selected_model,
            checkpoint=artifact_path,
            dataset_config=DATASET_CONFIG_PATH,
            model_config=selected_info.default_train_config,
        )
        predictions = run_predict(predict_args, dataframe=data)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.success("Forecast generated successfully.")
    st.dataframe(predictions.tail(20))
    if "timestamp" in predictions.columns and "y_hat" in predictions.columns:
        chart_df = predictions.set_index("timestamp")[["y_hat"]]
        if "y" in predictions.columns:
            chart_df["y"] = predictions["y"]
        chart_df.index = pd.to_datetime(chart_df.index)
        st.line_chart(chart_df, height=300)
