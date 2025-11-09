"""Phone-style live dashboard showing current usage and next-hour forecast."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.core.serving.live import (  # noqa: E402
    LiveSnapshot,
    list_meter_ids,
    load_xgb_live_snapshot,
    load_xgb_model,
)
from src.core.io import load_yaml  # noqa: E402
from core.data.preperation import load_aggregated_series, load_raw_series  # noqa: E402

MODEL_PATH = Path("models/xgboost/artifacts/model.json")
MODEL_CONFIG_PATH = Path("configs/model/xgboost.yaml")
DATASET_CONFIG_PATH = Path("configs/dataset.yaml")

st.set_page_config(page_title="Live Energy Forecast", layout="centered")


@st.cache_resource(show_spinner=False)
def _load_model_artifacts(path: Path):
    return load_xgb_model(path)


@st.cache_data(show_spinner=False)
def _load_configs() -> tuple[dict[str, object], dict[str, object]]:
    dataset_cfg = load_yaml(DATASET_CONFIG_PATH)
    model_cfg = load_yaml(MODEL_CONFIG_PATH)
    return dataset_cfg, model_cfg


@st.cache_data(show_spinner=False)
def _list_meters_cached() -> list[str]:
    return list_meter_ids(DATASET_CONFIG_PATH)


def _load_base_series(
    dataset_cfg: dict[str, object],
    model_cfg: dict[str, object],
    meter_id: str | None,
) -> pd.DataFrame:
    dataset_mode = str(model_cfg.get("dataset_mode", "aggregated")).lower()
    raw_df = pd.read_csv(dataset_cfg["raw_csv"])
    if dataset_mode == "raw":
        id_col = dataset_cfg.get("id_col")
        if not id_col:
            raise ValueError("dataset_id column required for raw mode")
        if meter_id is None:
            raise ValueError("Meter selection required for raw playback")
        raw_df[id_col] = raw_df[id_col].astype(str)
        raw_df = raw_df[raw_df[id_col] == str(meter_id)]
        base_series = load_raw_series(dataset_cfg, raw_df)
    else:
        base_series = load_aggregated_series(dataset_cfg, raw_df)
        meter_id = None
    ts_col = "timestamp"
    base_series[ts_col] = pd.to_datetime(base_series[ts_col])
    base_series = base_series.sort_values(ts_col).reset_index(drop=True)
    return base_series


def _build_playback_frames(
    model: XGBModel,
    feature_columns: Sequence[str],
    dataset_cfg: dict[str, object],
    model_cfg: dict[str, object],
    meter_id: str | None,
    max_frames: int,
) -> list[LiveSnapshot]:
    base_series = _load_base_series(dataset_cfg, model_cfg, meter_id)
    if base_series.empty:
        return []
    ts_col = "timestamp"
    timestamps = base_series[ts_col].tolist()
    if max_frames > 0:
        timestamps = timestamps[-max_frames:]
    frames: list[LiveSnapshot] = []
    for ts in timestamps:
        try:
            snapshot = load_xgb_live_snapshot(
                model=model,
                feature_columns=feature_columns,
                dataset_cfg=dataset_cfg,
                model_cfg=model_cfg,
                meter_id=meter_id,
                history_hours=24,
                focus_timestamp=ts,
            )
        except Exception:
            continue
        frames.append(snapshot)
    return frames


def _render_phone(snapshot: LiveSnapshot) -> None:
    delta = None
    if snapshot.previous_value is not None:
        delta = snapshot.current_value - snapshot.previous_value
    delta_text = "--"
    delta_class = "neutral"
    if delta is not None:
        delta_text = f"{delta:+.2f} kWh"
        if delta > 0:
            delta_class = "up"
        elif delta < 0:
            delta_class = "down"

    chips: list[str] = []
    if snapshot.dataset_mode == "raw" and snapshot.meter_id is not None:
        chips.append(f"Meter {snapshot.meter_id}")
    if snapshot.flags.get("is_public_holiday"):
        chips.append("Public holiday")
    if snapshot.flags.get("is_school_holiday"):
        chips.append("School holiday")
    if snapshot.current_timestamp.weekday() >= 5:
        chips.append("Weekend")

    chip_html = "".join(f"<span class='chip'>{text}</span>" for text in chips) or "<span class='chip muted'>No special tags</span>"

    warning_html = ""
    if snapshot.current_over_threshold or snapshot.forecast_over_threshold:
        status = []
        if snapshot.current_over_threshold:
            status.append("current")
        if snapshot.forecast_over_threshold:
            status.append("next-hour")
        label = " and ".join(status)
        warning_html = (
            f"<div class='warning-banner'>&#9888;&#65039; High usage ({label}) above "
            f"{snapshot.usage_threshold:.1f} kWh</div>"
        )

    phone_html = f"""
    <div class="phone-shell">
        {warning_html}
        <div class="phone-header">
            <div class="phone-title">Current usage</div>
            <div class="phone-timestamp">{snapshot.current_timestamp.strftime("%a %d %b | %H:%M")}</div>
        </div>
        <div class="current-value">{snapshot.current_value:.2f}<span class="unit">kWh</span></div>
        <div class="delta {delta_class}">{delta_text}</div>
        <div class="chip-row">{chip_html}</div>
        <div class="forecast-card">
            <div class="forecast-label">Next hour forecast</div>
            <div class="forecast-value">{snapshot.forecast_value:.2f}<span class="unit">kWh</span></div>
            <div class="forecast-time">{snapshot.forecast_timestamp.strftime("%a %d %b | %H:%M")}</div>
        </div>
        <div class="cost-card">
            <div class="cost-heading">Cost summary</div>
            <div class="cost-row"><span>Today's cost</span><span>${snapshot.today_cost:,.2f}</span></div>
            <div class="cost-row"><span>Current hour</span><span>${snapshot.current_cost:,.2f}</span></div>
            <div class="cost-row"><span>Next hour forecast</span><span>${snapshot.forecast_cost:,.2f}</span></div>
            <div class="cost-note">@ ${snapshot.cost_per_kwh:.2f} per kWh</div>
        </div>
    </div>
    """
    full_html = f"{PHONE_CSS}{phone_html}"
    st_html(full_html, height=620, scrolling=False)


def _render_history(snapshot: LiveSnapshot) -> None:
    history_df = snapshot.history.copy()
    history_df["Actual"] = history_df["Actual"].astype(float)
    history_df["Forecast"] = history_df["Forecast"].astype(float)

    chart = (
        alt.Chart(history_df)
        .transform_fold(
            ["Actual", "Forecast"],
            as_=["Series", "Value"],
        )
        .mark_line(point=True, size=2)
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("Value:Q", title="kWh"),
            color=alt.Color("Series:N", scale=alt.Scale(range=["#4cc9f0", "#f72585"])),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Timestamp"),
            alt.Tooltip("Series:N"),
            alt.Tooltip("Value:Q", format=".2f", title="kWh"),
        ],
    )
        .properties(width=720, height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_feature_summary(snapshot: LiveSnapshot) -> None:
    features = snapshot.future_features.sort_index()
    df = pd.DataFrame({"Feature": features.index, "Value": features.values})
    st.caption("Feature vector for next hour")
    st.dataframe(df, width="stretch", hide_index=True)



PHONE_CSS = """
<style>
.warning-banner {
    margin-bottom: 0.8rem;
    padding: 0.6rem 0.8rem;
    border-radius: 14px;
    background: rgba(251, 191, 36, 0.15);
    color: #fbbf24;
    text-align: center;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.phone-shell {
    margin: 1.5rem auto;
    width: 340px;
    padding: 1.75rem 1.5rem;
    border-radius: 32px;
    background: linear-gradient(145deg, #0f172a, #111827);
    box-shadow: 0 30px 60px rgba(15, 23, 42, 0.45);
    color: #f8fafc;
    font-family: 'Segoe UI', sans-serif;
}
.phone-header {
    text-align: center;
    margin-bottom: 1rem;
}
.phone-title {
    font-size: 0.95rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.2rem;
}
.phone-timestamp {
    font-size: 0.85rem;
    color: rgba(248, 250, 252, 0.7);
}
.current-value {
    font-size: 3.4rem;
    font-weight: 600;
    text-align: center;
    line-height: 1.1;
}
.current-value .unit {
    font-size: 1.2rem;
    margin-left: 0.35rem;
    font-weight: 400;
    color: rgba(248, 250, 252, 0.65);
}
.delta {
    text-align: center;
    font-size: 0.95rem;
    margin-bottom: 0.8rem;
    font-weight: 500;
}
.delta.up { color: #22c55e; }
.delta.down { color: #f97316; }
.delta.neutral { color: rgba(248, 250, 252, 0.6); }
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    justify-content: center;
    margin-bottom: 1.2rem;
}
.chip {
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.chip.muted {
    background: rgba(148, 163, 184, 0.12);
    color: rgba(226, 232, 240, 0.6);
}
.forecast-card {
    margin-top: 0.5rem;
    padding: 1rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(30, 64, 175, 0.7), rgba(56, 189, 248, 0.35));
    text-align: center;
}
.forecast-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(248, 250, 252, 0.8);
}
.forecast-value {
    font-size: 2.1rem;
    font-weight: 600;
    margin: 0.4rem 0;
}
.forecast-value .unit {
    font-size: 1rem;
    margin-left: 0.25rem;
    color: rgba(248, 250, 252, 0.7);
    font-weight: 400;
}
.forecast-time {
    font-size: 0.82rem;
    color: rgba(248, 250, 252, 0.75);
}
.cost-card {
    margin-top: 1rem;
    padding: 1rem;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(148, 163, 184, 0.2);
}
.cost-heading {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(248, 250, 252, 0.75);
    margin-bottom: 0.5rem;
    text-align: center;
}
.cost-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.95rem;
    margin: 0.15rem 0;
    color: rgba(248, 250, 252, 0.9);
}
.cost-row span:last-child {
    font-weight: 600;
}
.cost-note {
    margin-top: 0.6rem;
    font-size: 0.75rem;
    text-align: center;
    color: rgba(226, 232, 240, 0.6);
}
</style>
"""



def main() -> None:
    st.title("Live Energy Dashboard")
    st.caption("Mobile-style view of the latest usage and one-hour-ahead forecast.")

    if not MODEL_PATH.exists():
        st.error(f"Model checkpoint not found at {MODEL_PATH}. Train the XGBoost model first.")
        st.stop()

    try:
        model, feature_columns = _load_model_artifacts(MODEL_PATH)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    dataset_cfg, model_cfg = _load_configs()
    dataset_mode = str(model_cfg.get("dataset_mode", "aggregated")).lower()

    meter_id: str | None = None
    if dataset_mode == "raw":
        meter_options = _list_meters_cached()
        if not meter_options:
            st.error("No meters found in the dataset; cannot generate live view.")
            st.stop()
        meter_id = st.sidebar.selectbox("Meter", meter_options, help="Select the meter to visualise.")
        st.sidebar.caption("Dataset mode: raw per-meter series.")
    else:
        st.sidebar.caption("Dataset mode: aggregated mean profile.")

    st.sidebar.info("Click **Refresh** to recompute cached data using the latest saved artefacts.")
    if st.sidebar.button("Refresh"):
        st.cache_data.clear()
        for key in ("playback_frames", "playback_cache_key", "playhead", "timeline_slider", "play_mode"):
            st.session_state.pop(key, None)
        st.rerun()

    playback_window = st.sidebar.slider("Playback window (hours)", 12, 168, 48, step=6)
    playback_key = (dataset_mode, meter_id if meter_id is not None else "ALL", playback_window)
    if st.session_state.get("playback_cache_key") != playback_key:
        frames = _build_playback_frames(
            model=model,
            feature_columns=feature_columns,
            dataset_cfg=dataset_cfg,
            model_cfg=model_cfg,
            meter_id=meter_id,
            max_frames=playback_window,
        )
        st.session_state["playback_frames"] = frames
        st.session_state["playback_cache_key"] = playback_key
        st.session_state["playhead"] = max(len(frames) - 1, 0)
    else:
        frames = st.session_state.get("playback_frames", [])

    if frames:
        if not hasattr(frames[0], "today_cost"):
            st.session_state.pop("playback_frames", None)
            st.session_state.pop("playback_cache_key", None)
            st.rerun()
        st.session_state.setdefault("playhead", len(frames) - 1)
        st.session_state["timeline_slider"] = st.session_state["playhead"]

    play_speed = st.sidebar.slider("Playback speed (ms)", 200, 2000, 800, step=100)
    auto_play = st.sidebar.checkbox(
        "Auto-play timeline",
        value=st.session_state.get("play_mode", False),
        key="auto_play_checkbox",
    )
    st.session_state["play_mode"] = auto_play

    if frames:
        st.sidebar.caption(f"Playback frames loaded: {len(frames)}")
        ctrl_prev, ctrl_next = st.sidebar.columns(2)
        if ctrl_prev.button("Prev"):
            st.session_state["play_mode"] = False
            st.session_state["playhead"] = max(0, st.session_state["playhead"] - 1)
            st.session_state["timeline_slider"] = st.session_state["playhead"]
        if ctrl_next.button("Next"):
            st.session_state["play_mode"] = False
            st.session_state["playhead"] = min(len(frames) - 1, st.session_state["playhead"] + 1)
            st.session_state["timeline_slider"] = st.session_state["playhead"]

        slider_kwargs = {
            "label": "Timeline index",
            "min_value": 0,
            "max_value": len(frames) - 1,
            "key": "timeline_slider",
        }
        if "timeline_slider" not in st.session_state:
            slider_kwargs["value"] = st.session_state["playhead"]
        slider_val = st.sidebar.slider(**slider_kwargs)
        if slider_val != st.session_state["playhead"]:
            st.session_state["play_mode"] = False
            st.session_state["playhead"] = slider_val

        if st.session_state.get("play_mode", False):
            time.sleep(play_speed / 1000.0)
            st.session_state["playhead"] = (st.session_state["playhead"] + 1) % len(frames)
            st.session_state["timeline_slider"] = st.session_state["playhead"]
            st.rerun()

    if frames:
        snapshot = frames[st.session_state["playhead"]]
    else:
        try:
            snapshot = load_xgb_live_snapshot(
                model=model,
                feature_columns=feature_columns,
                dataset_cfg=dataset_cfg,
                model_cfg=model_cfg,
                meter_id=meter_id,
                history_hours=24,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to compute live forecast: {exc}")
            st.stop()

    _render_phone(snapshot)

    st.caption(f"Timeline view: {snapshot.current_timestamp} -> {snapshot.forecast_timestamp}")

    st.subheader("Recent history")
    _render_history(snapshot)

    with st.expander("Feature vector (next hour)"):
        _render_feature_summary(snapshot)

    st.caption(
        "Data source: configs/dataset.yaml | Model config: configs/model/xgboost.yaml "
        f"| Dataset mode: {snapshot.dataset_mode}"
    )


if __name__ == "__main__":
    main()
