"""Per-meter exploratory data analysis utilities for merged electricity / weather data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "datasets" / "merged_electricity_weather.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
METER_COL = "meter_ui"
TIMESTAMP_COL = "timestamp"


def load_meter_dataset(
    data_path: Path = DEFAULT_DATA_PATH,
    use_optimized_dtypes: bool = True,
) -> pd.DataFrame:
    """Load the merged CSV and append a timestamp column for temporal analysis."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    dtype_map = None
    if use_optimized_dtypes:
        dtype_map = {
            "nmi_ui": "category",
            "meter_ui": "category",
            "date": "string",
            "time": "string",
            "error_check_day": "float32",
            "error_check_hour": "float32",
            "delivered_value": "float32",
            "daily_energy_usage": "float32",
            "received_value": "float32",
            "power_zero": "float32",
            "daily_energy_zero": "float32",
            "rainfall_amount_millimetres": "float32",
            "period_over_which_rainfall_was_measured_days": "float32",
            "maximum_temperature_degree_c": "float32",
            "days_of_accumulation_of_maximum_temperature": "float32",
            "minimum_temperature_degree_c": "float32",
            "days_of_accumulation_of_minimum_temperature": "float32",
            "daily_global_solar_exposure_mj_m_m": "float32",
        }

    df = pd.read_csv(
        data_path,
        dtype=dtype_map,
        parse_dates=["aggregate_date"],
        na_values=["", " "],
    )
    date_str = df["date"].astype(str).str.strip()
    time_str = df["time"].astype(str).str.strip()
    df[TIMESTAMP_COL] = pd.to_datetime(
        date_str + " " + time_str,
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    missing_timestamp = df[TIMESTAMP_COL].isna()
    if missing_timestamp.any():
        df.loc[missing_timestamp, TIMESTAMP_COL] = df.loc[
            missing_timestamp, "aggregate_date"
        ].dt.tz_localize(None)
    df = df.sort_values([METER_COL, TIMESTAMP_COL]).reset_index(drop=True)

    # Force meter IDs to category even when dtype_map is disabled.
    df[METER_COL] = df[METER_COL].astype("category")
    df["nmi_ui"] = df["nmi_ui"].astype("category")

    return df


def compute_meter_summaries(
    df: pd.DataFrame,
    meter_col: str = METER_COL,
    timestamp_col: str = TIMESTAMP_COL,
) -> pd.DataFrame:
    """Aggregate high level statistics for every meter."""
    grouped = df.groupby(meter_col, observed=True)

    summary = grouped.agg(
        nmi_count=("nmi_ui", "nunique"),
        observation_count=("delivered_value", "count"),
        start_time=(timestamp_col, "min"),
        end_time=(timestamp_col, "max"),
        delivered_mean=("delivered_value", "mean"),
        delivered_median=("delivered_value", "median"),
        delivered_std=("delivered_value", "std"),
        delivered_min=("delivered_value", "min"),
        delivered_p25=("delivered_value", lambda x: x.quantile(0.25)),
        delivered_p75=("delivered_value", lambda x: x.quantile(0.75)),
        delivered_max=("delivered_value", "max"),
        delivered_sum=("delivered_value", "sum"),
        daily_energy_mean=("daily_energy_usage", "mean"),
        daily_energy_std=("daily_energy_usage", "std"),
        daily_energy_sum=("daily_energy_usage", "sum"),
        received_mean=("received_value", "mean"),
        received_sum=("received_value", "sum"),
        power_zero_rate=("power_zero", "mean"),
        daily_energy_zero_rate=("daily_energy_zero", "mean"),
    ).astype(
        {
            "nmi_count": "int64",
            "observation_count": "int64",
        }
    )

    summary["period_hours"] = (
        summary["end_time"] - summary["start_time"]
    ).dt.total_seconds() / 3600.0 + 1
    summary["period_days"] = summary["period_hours"] / 24.0
    summary["coverage_pct"] = (
        summary["observation_count"] / summary["period_hours"].clip(lower=1.0) * 100.0
    )
    summary["delivered_cv"] = (
        summary["delivered_std"] / summary["delivered_mean"]
    ).replace([np.inf, -np.inf], np.nan)
    summary["power_zero_pct"] = summary.pop("power_zero_rate") * 100.0
    summary["daily_energy_zero_pct"] = summary.pop("daily_energy_zero_rate") * 100.0

    return summary


def attach_peer_comparison(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    meter_col: str = METER_COL,
) -> pd.DataFrame:
    """Add mean comparison columns against the pooled peers."""
    delivered_stats = df.groupby(meter_col, observed=True)["delivered_value"].agg(
        ["sum", "count"]
    )
    energy_stats = df.groupby(meter_col, observed=True)["daily_energy_usage"].agg(
        ["sum", "count"]
    )

    delivered_totals = delivered_stats["sum"]
    delivered_counts = delivered_stats["count"]
    all_delivered_sum = delivered_totals.sum()
    all_delivered_count = delivered_counts.sum()

    energy_totals = energy_stats["sum"]
    energy_counts = energy_stats["count"]
    all_energy_sum = energy_totals.sum()
    all_energy_count = energy_counts.sum()

    peer_delivered_mean = (all_delivered_sum - delivered_totals) / (
        all_delivered_count - delivered_counts
    )
    peer_energy_mean = (all_energy_sum - energy_totals) / (
        all_energy_count - energy_counts
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        summary["delivered_vs_peers_pct"] = (
            (summary["delivered_mean"] - peer_delivered_mean)
            / peer_delivered_mean
            * 100.0
        )
        summary["daily_energy_vs_peers_pct"] = (
            (summary["daily_energy_mean"] - peer_energy_mean)
            / peer_energy_mean
            * 100.0
        )

    summary["delivered_mean_rank"] = (
        summary["delivered_mean"].rank(ascending=False, method="dense").astype(int)
    )
    summary["delivered_sum_rank"] = (
        summary["delivered_sum"].rank(ascending=False, method="dense").astype(int)
    )

    return summary


def hourly_usage_profile(
    df: pd.DataFrame,
    meter_id: str,
    meter_col: str = METER_COL,
    timestamp_col: str = TIMESTAMP_COL,
    value_col: str = "delivered_value",
) -> pd.DataFrame:
    """Return hourly mean usage for a meter alongside peer averages."""
    meter_mask = df[meter_col] == meter_id
    if not meter_mask.any():
        raise ValueError(f"Meter {meter_id} not present in dataset.")

    hours = df[timestamp_col].dt.hour
    meter_profile = (
        df.loc[meter_mask].groupby(hours[meter_mask])[value_col].agg(["mean", "median"])
    )
    peers_profile = (
        df.loc[~meter_mask].groupby(hours[~meter_mask])[value_col].mean()
    )

    profile = (
        meter_profile.rename(columns={"mean": "meter_mean", "median": "meter_median"})
        .reindex(range(24))
        .join(peers_profile.rename("peer_mean"), how="left")
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        profile["lift_pct"] = (
            (profile["meter_mean"] - profile["peer_mean"])
            / profile["peer_mean"]
            * 100.0
        )

    profile.index.name = "hour"
    return profile.reset_index()


def monthly_usage_profile(
    df: pd.DataFrame,
    meter_id: str,
    meter_col: str = METER_COL,
    timestamp_col: str = TIMESTAMP_COL,
    value_col: str = "delivered_value",
) -> pd.DataFrame:
    """Summarise monthly usage for a meter vs peers."""
    df = df.copy()
    df["month"] = df[timestamp_col].dt.to_period("M")

    meter_mask = df[meter_col] == meter_id
    if not meter_mask.any():
        raise ValueError(f"Meter {meter_id} not present in dataset.")

    meter_month = df.loc[meter_mask].groupby("month")[value_col].agg(
        meter_mean="mean",
        meter_median="median",
        meter_sum="sum",
    )
    peers_month = df.loc[~meter_mask].groupby("month")[value_col].agg(
        peer_mean="mean",
        peer_median="median",
        peer_sum="sum",
    )
    profile = meter_month.join(peers_month, how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        profile["sum_contribution_pct"] = (
            profile["meter_sum"] / (profile["meter_sum"] + profile["peer_sum"]) * 100.0
        )

    return profile.reset_index()


def _set_plot_theme() -> None:
    """Apply a consistent visual style regardless of seaborn availability."""
    if sns is not None:
        sns.set_theme(style="whitegrid")
        return

    available = set(plt.style.available)
    for candidate in (
        "seaborn-v0_8-whitegrid",
        "seaborn-whitegrid",
        "ggplot",
        "fast",
    ):
        if candidate in available:
            plt.style.use(candidate)
            return


def plot_hourly_profile(
    profile: pd.DataFrame,
    meter_id: str,
    value_col_meter: str = "meter_mean",
    value_col_peer: str = "peer_mean",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Create a line plot comparing hourly mean usage."""
    _set_plot_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(profile["hour"], profile[value_col_meter], label=f"{meter_id} mean")
    ax.plot(profile["hour"], profile[value_col_peer], label="Peer mean", linestyle="--")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Delivered value (kWh)")
    ax.set_title(f"Hourly delivered value profile - {meter_id}")
    ax.legend()
    fig.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{meter_id}_hourly_profile.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    plt.show()
    return None


def build_meter_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    meter_id: str,
    output_dir: Optional[Path] = None,
    export_profiles: bool = False,
    plot_profiles: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Compile per-meter tables and optionally export / plot them."""
    if meter_id not in summary.index:
        raise ValueError(f"Meter {meter_id} not present in summary.")

    meter_summary = summary.loc[[meter_id]].copy()
    hourly_profile = hourly_usage_profile(df, meter_id)
    monthly_profile = monthly_usage_profile(df, meter_id)

    results = {
        "summary": meter_summary,
        "hourly_profile": hourly_profile,
        "monthly_profile": monthly_profile,
    }

    if export_profiles:
        export_dir = output_dir or DEFAULT_OUTPUT_DIR
        export_dir.mkdir(parents=True, exist_ok=True)
        for name, table in results.items():
            export_path = export_dir / f"{meter_id}_{name}.csv"
            table.to_csv(export_path, index=False)

    if plot_profiles:
        plot_hourly_profile(
            hourly_profile,
            meter_id=meter_id,
            output_dir=output_dir if export_profiles else None,
        )

    return results


def _finalize_plot(fig: plt.Figure, output_path: Optional[Path], show: bool) -> Optional[Path]:
    """Centralise saving/showing behaviour to keep plotting helpers tidy."""
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def _plot_top_bar(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    top_n: int,
    output_path: Optional[Path],
    show: bool,
) -> Optional[Path]:
    _set_plot_theme()
    ordered = summary.dropna(subset=[metric]).sort_values(metric, ascending=False).head(top_n)
    if ordered.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    meters = ordered.index.astype(str)
    values = ordered[metric]

    color = None
    if sns is not None:
        color = sns.color_palette("viridis", len(meters))

    ax.barh(meters, values, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Meter")
    ax.invert_yaxis()

    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value:,.2f}", va="center", ha="left", fontsize=8)

    fig.tight_layout()
    return _finalize_plot(fig, output_path, show)


def _plot_metric_histogram(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    output_path: Optional[Path],
    show: bool,
) -> Optional[Path]:
    _set_plot_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    data = summary[metric].dropna()

    if sns is not None:
        sns.histplot(data, bins=20, kde=True, ax=ax)
    else:
        ax.hist(data, bins=20, alpha=0.75, color="tab:blue", edgecolor="black")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return _finalize_plot(fig, output_path, show)


def _plot_metric_scatter(
    summary: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    annotate_top: int,
    output_path: Optional[Path],
    show: bool,
) -> Optional[Path]:
    _set_plot_theme()
    fig, ax = plt.subplots(figsize=(8, 6))

    x = summary[x_metric]
    y = summary[y_metric]
    ax.scatter(x, y, alpha=0.7, color="tab:purple")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if annotate_top > 0:
        top_candidates = summary.nlargest(annotate_top, y_metric)
        for meter, row in top_candidates.iterrows():
            ax.annotate(
                str(meter),
                (row[x_metric], row[y_metric]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    fig.tight_layout()
    return _finalize_plot(fig, output_path, show)


def generate_summary_plots(
    summary: pd.DataFrame,
    output_dir: Optional[Path] = None,
    show: bool = False,
    top_n: int = 15,
) -> List[Path]:
    """Produce a standard set of cross-meter plots to spot outliers quickly."""
    if summary.empty:
        return []

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    top_n = max(1, min(top_n, len(summary)))
    saved_paths: List[Path] = []

    outputs = [
        (
            _plot_top_bar,
            {
                "metric": "delivered_mean",
                "title": f"Top {top_n} meters by average delivered value",
                "xlabel": "Average delivered value (kWh)",
            },
            f"top_{top_n}_delivered_mean.png",
        ),
        (
            _plot_top_bar,
            {
                "metric": "delivered_sum",
                "title": f"Top {top_n} meters by total delivered value",
                "xlabel": "Total delivered value (kWh)",
            },
            f"top_{top_n}_delivered_sum.png",
        ),
        (
            _plot_metric_histogram,
            {
                "metric": "power_zero_pct",
                "title": "Distribution of zero-power readings across meters",
                "xlabel": "Power zero percentage",
            },
            "power_zero_distribution.png",
        ),
        (
            _plot_metric_scatter,
            {
                "x_metric": "delivered_mean",
                "y_metric": "power_zero_pct",
                "title": "Delivered mean vs zero-power percentage",
                "xlabel": "Average delivered value (kWh)",
                "ylabel": "Power zero percentage",
                "annotate_top": min(5, len(summary)),
            },
            "delivered_mean_vs_zero_pct.png",
        ),
        (
            _plot_metric_histogram,
            {
                "metric": "coverage_pct",
                "title": "Coverage percentage per meter",
                "xlabel": "Coverage percentage",
            },
            "coverage_distribution.png",
        ),
    ]

    for func, kwargs, filename in outputs:
        output_path = output_dir / filename if output_dir is not None else None
        call_kwargs = {
            "summary": summary,
            "output_path": output_path,
            "show": show,
            **kwargs,
        }
        if "top_n" in func.__code__.co_varnames:
            call_kwargs["top_n"] = top_n
        result = func(**call_kwargs)
        if isinstance(result, Path):
            saved_paths.append(result)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-meter exploratory statistics and optionally focus on a"
            " single meter for deeper inspection."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to merged CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--meter",
        type=str,
        help="Meter UI to inspect in detail (e.g. 'M1').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for exported tables/figures.",
    )
    parser.add_argument(
        "--export-summary",
        action="store_true",
        help="Write the per-meter summary table to CSV.",
    )
    parser.add_argument(
        "--export-profiles",
        action="store_true",
        help="When --meter is set, export the meter-specific tables.",
    )
    parser.add_argument(
        "--plot-profiles",
        action="store_true",
        help="Show (or save) comparison plots for the selected meter.",
    )
    parser.add_argument(
        "--summary-plots",
        action="store_true",
        help="Display cross-meter comparison plots.",
    )
    parser.add_argument(
        "--export-summary-plots",
        action="store_true",
        help="Save cross-meter plots to --output-dir.",
    )
    parser.add_argument(
        "--summary-top-n",
        type=int,
        default=15,
        help="Number of meters to display in ranked summary charts (default: 15).",
    )
    parser.add_argument(
        "--no-optimized-dtypes",
        action="store_true",
        help="Disable dtype optimisation on load (falls back to pandas defaults).",
    )
    return parser.parse_args()


def main() -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
    args = parse_args()
    df = load_meter_dataset(
        data_path=args.data_path,
        use_optimized_dtypes=not args.no_optimized_dtypes,
    )
    summary = compute_meter_summaries(df)
    summary = attach_peer_comparison(df, summary)

    if args.export_summary:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "per_meter_summary.csv"
        summary.to_csv(summary_path)
        print(f"Exported per-meter summary to {summary_path}")

    print(
        summary[
            [
                "observation_count",
                "delivered_mean",
                "delivered_vs_peers_pct",
                "power_zero_pct",
                "coverage_pct",
            ]
        ]
        .sort_values("delivered_mean", ascending=False)
        .head(10)
    )

    if args.summary_plots or args.export_summary_plots:
        output_dir = args.output_dir if args.export_summary_plots else None
        saved = generate_summary_plots(
            summary,
            output_dir=output_dir,
            show=args.summary_plots,
            top_n=max(1, args.summary_top_n),
        )
        if saved:
            print("\nSaved summary plots:")
            for path in saved:
                print(f" - {path}")

    meter_outputs: Optional[Dict[str, pd.DataFrame]] = None
    if args.meter:
        meter_outputs = build_meter_report(
            df,
            summary,
            meter_id=args.meter,
            output_dir=args.output_dir,
            export_profiles=args.export_profiles,
            plot_profiles=args.plot_profiles,
        )
        print(f"\nSummary for meter {args.meter}:")
        print(meter_outputs["summary"].T)
        print("\nHourly profile (first five rows):")
        print(meter_outputs["hourly_profile"].head())

    return summary, meter_outputs


if __name__ == "__main__":
    main()
