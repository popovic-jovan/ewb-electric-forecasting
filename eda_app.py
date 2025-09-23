
import os
import sys
import importlib
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Electricity Usage EDA", layout="wide")

st.title("⚡ Electricity Usage EDA — Interactive App")
st.caption("Scroll, filter, zoom, and export — designed for fast feature exploration before modelling.")

# -------------------------
# Utilities
# -------------------------
@st.cache_data(show_spinner=False)
def read_parquet_or_excel(file) -> pd.DataFrame:
    name = getattr(file, "name", str(file))
    try:
        if name.lower().endswith((".parquet", ".pq")):
            return pd.read_parquet(file)
        elif name.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(file)
        elif name.lower().endswith(".csv"):
            return pd.read_csv(file)
        else:
            # try sniff parquet first
            try:
                return pd.read_parquet(file)
            except Exception:
                file.seek(0)
                try:
                    return pd.read_excel(file)
                except Exception:
                    file.seek(0)
                    return pd.read_csv(file)
    finally:
        try:
            file.seek(0)
        except Exception:
            pass

"""Try to import data_loader.py and read the global df variable."""
def safe_import_data_loader() -> Optional[pd.DataFrame]:
    try:
        if "" not in sys.path:
            sys.path.insert(0, "")
        if "/mnt/data" not in sys.path:
            sys.path.insert(0, "/mnt/data")
        mod = importlib.import_module("data_loader")
        df = getattr(mod, "df", None)
        if isinstance(df, pd.DataFrame):
            return df.copy()
        else:
            return None
    except Exception as e:
        st.info(f"Could not import `data_loader.py`: {e}")
        return None

def detect_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    # Use pandas type checks to handle extension dtypes (e.g., Int64)
    from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

    datetime_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c]) and c not in datetime_cols]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + datetime_cols]
    return numeric_cols, categorical_cols, datetime_cols

def human_readable_counts(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    mem_mb = df.memory_usage(index=True).sum() / (1024**2)
    return f"{rows:,} rows × {cols:,} columns • ~{mem_mb:,.2f} MB in RAM"

# -------------------------
# Data source
# -------------------------
st.sidebar.header("📦 Data Source")
source = st.sidebar.radio("Choose how to load data", ["Use data_loader.py", "Upload file"], index=0)

df: Optional[pd.DataFrame] = None

if source == "Use data_loader.py":
    df = safe_import_data_loader()
    if df is None:
        st.warning("`data_loader.py` not found or no global `df` available. Switch to **Upload file**.")
else:
    uploaded = st.sidebar.file_uploader("Upload a Parquet, Excel, or CSV", type=["parquet", "pq", "xlsx", "xls", "csv"])
    if uploaded:
        df = read_parquet_or_excel(uploaded)

if df is None:
    st.stop()

st.success(f"Loaded dataset — {human_readable_counts(df)}")
st.write("Preview:", df.head())

# -------------------------
# Column selection
# -------------------------
st.sidebar.header("🧰 Column Selection")
numeric_cols, categorical_cols, datetime_cols = detect_types(df)

# Drop obvious ID-like columns if user opts in
drop_id_like = st.sidebar.checkbox("Auto-drop ID-like columns (e.g., *_id, id, meter, household)", value=False)
if drop_id_like:
    id_like = [c for c in df.columns if c.lower() in {"id"} or c.lower().endswith("_id") or "meter" in c.lower() or "household" in c.lower()]
    kept = [c for c in df.columns if c not in id_like]
    if kept:
        df = df[kept]
        numeric_cols, categorical_cols, datetime_cols = detect_types(df)
        st.sidebar.caption(f"Dropped: {', '.join(id_like) or 'None'}")

# Let user pick columns to include
cols_selected = st.sidebar.multiselect("Columns to include", options=list(df.columns), default=list(df.columns))
if cols_selected:
    df = df[cols_selected]
    numeric_cols, categorical_cols, datetime_cols = detect_types(df)

# Target selection (optional)
target = st.sidebar.selectbox("Target column (for scatter/feature tools)", options=[None] + list(df.columns), index=0)

# -------------------------
# Filters
# -------------------------
st.sidebar.header("🔎 Filters")
with st.sidebar.expander("Add filters", expanded=False):
    # Categorical filters
    for col in categorical_cols:
        vals = sorted([v for v in df[col].dropna().unique().tolist() if isinstance(v, (str, int, float, bool))])
        selected = st.multiselect(f"{col}", vals, default=[])
        if selected:
            df = df[df[col].isin(selected)]

    # Numeric range filters
    for col in numeric_cols:
        # Use pandas min/max to support nullable dtypes (e.g., Int64)
        try:
            col_min = float(df[col].min(skipna=True))
            col_max = float(df[col].max(skipna=True))
        except Exception:
            # Fallback to numpy if needed
            col_min, col_max = float(np.nanmin(df[col].to_numpy(dtype=float, na_value=np.nan))), float(np.nanmax(df[col].to_numpy(dtype=float, na_value=np.nan)))
        if np.isfinite(col_min) and np.isfinite(col_max) and col_min != col_max:
            rng = st.slider(f"{col} range", min_value=col_min, max_value=col_max, value=(col_min, col_max))
            df = df[(df[col] >= rng[0]) & (df[col] <= rng[1])]

    # Datetime range filters
    for col in datetime_cols:
        min_dt, max_dt = pd.to_datetime(df[col].min()), pd.to_datetime(df[col].max())
        if pd.notna(min_dt) and pd.notna(max_dt) and min_dt != max_dt:
            start, end = st.date_input(f"{col} date range", (min_dt.date(), max_dt.date()))
            if isinstance(start, tuple):
                # streamlit older versions return tuple on multi-date input
                start, end = start
            df = df[(df[col] >= pd.to_datetime(start)) & (df[col] <= pd.to_datetime(end))]

st.info(f"Active dataset after filters — {human_readable_counts(df)}")

# -------------------------
# Tabs for analysis
# -------------------------
tab_overview, tab_dist, tab_scatter, tab_box, tab_corr, tab_time, tab_group, tab_importance = st.tabs([
    "Overview", "Distributions", "Scatter vs Target", "Box by Category", "Correlation", "Time series", "Group & Aggregate", "Feature importance"
])

with tab_overview:
    st.subheader("Dataset Overview")
    st.dataframe(df.head(50), use_container_width=True)
    st.markdown("### Missing Values")
    miss = df.isna().sum().sort_values(ascending=False)
    st.write(miss[miss > 0])

with tab_dist:
    st.subheader("Univariate Distributions")
    if len(df.columns) == 0:
        st.write("No columns selected")
    else:
        cols = st.multiselect("Columns to plot", options=list(df.columns), default=numeric_cols[:5] if numeric_cols else list(df.columns)[:5])
        bins = st.slider("Bins (for histograms)", min_value=10, max_value=200, value=40)
        for col in cols:
            if col in numeric_cols:
                fig = px.histogram(df, x=col, nbins=bins, marginal="box", title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # categorical
                vc = df[col].value_counts(dropna=False).reset_index()
                vc.columns = [col, "count"]
                fig = px.bar(vc, x=col, y="count", title=f"Counts of {col}")
                st.plotly_chart(fig, use_container_width=True)

with tab_scatter:
    st.subheader("Scatter vs Target")
    if target is None:
        st.warning("Select a target column in the sidebar to enable this tab.")
    else:
        xcol = st.selectbox("X-axis (numeric)", options=numeric_cols)
        trend = st.checkbox("Show LOWESS trend (approx)", value=False)
        fig = px.scatter(df, x=xcol, y=target, trendline="lowess" if trend else None, title=f"{target} vs {xcol}")
        st.plotly_chart(fig, use_container_width=True)

with tab_box:
    st.subheader("Boxplots by Category")
    if target is None:
        st.warning("Select a target column to plot distributions by category.")
    else:
        cat = st.selectbox("Category", options=categorical_cols)
        fig = px.box(df, x=cat, y=target, points="all", title=f"{target} by {cat}")
        st.plotly_chart(fig, use_container_width=True)

with tab_corr:
    st.subheader("Correlation Heatmap (numeric only)")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        # Top correlations with target
        if target in numeric_cols:
            st.markdown("#### Top correlations with target")
            top = corr[target].drop(labels=[target]).sort_values(ascending=False)
            st.write(top.head(10))
    else:
        st.info("Need at least 2 numeric columns.")

with tab_time:
    st.subheader("Time Series")
    if datetime_cols:
        tcol = st.selectbox("Datetime column", options=datetime_cols)
        ycol = st.selectbox("Y value", options=numeric_cols if numeric_cols else [])
        freq = st.selectbox("Aggregation", options=["None (raw)", "H", "D", "W", "M", "Q", "Y"], index=2)
        if ycol:
            _df = df[[tcol, ycol]].dropna()
            if freq != "None (raw)":
                _df = _df.set_index(tcol).resample(freq).mean(numeric_only=True).reset_index()
            fig = px.line(_df, x=tcol, y=ycol, title=f"{ycol} over time ({freq})")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No datetime columns detected.")

with tab_group:
    st.subheader("Group & Aggregate")
    if categorical_cols and numeric_cols:
        gcol = st.selectbox("Group by (category)", options=categorical_cols)
        ycol = st.selectbox("Aggregate numeric column", options=numeric_cols, key="group_ycol")
        agg = st.selectbox("Aggregation", options=["mean", "sum", "median", "min", "max", "count"], index=0)
        grouped = df.groupby(gcol, dropna=False)[ycol].agg(agg).reset_index().sort_values(ycol if agg!="count" else 0, ascending=False)
        fig = px.bar(grouped, x=gcol, y=ycol if agg!="count" else grouped.columns[-1], title=f"{agg.upper()} of {ycol} by {gcol}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(grouped, use_container_width=True)
    else:
        st.info("Need at least one categorical and one numeric column.")

with tab_importance:
    st.subheader("Quick Feature Importance (Random Forest)")
    st.caption("This is a rough, fast estimate. Proper ML should be done in a notebook/pipeline.")
    if target is None:
        st.warning("Select a numeric target in the sidebar.")
    elif target not in numeric_cols:
        st.warning("Feature importance here requires a numeric target.")
    else:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        # Prepare X/y
        X = df.drop(columns=[target]).copy()
        y = df[target].copy()

        # One-hot encode categoricals
        X = pd.get_dummies(X, drop_first=True)

        # Drop columns with all-NaN
        X = X.dropna(axis=1, how="all")

        # Simple imputation
        X = X.fillna(X.median(numeric_only=True))
        y = y.fillna(y.median())

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig = px.bar(importances.head(25)[::-1], orientation="h", title=f"Top Features (R² = {r2:.3f})")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Show full importance table"):
                st.dataframe(importances.rename("importance"), use_container_width=True)
        except Exception as e:
            st.error(f"Could not compute importance: {e}")

# -------------------------
# Download section
# -------------------------
st.sidebar.header("⬇️ Download")
fmt = st.sidebar.selectbox("Format", ["CSV", "Parquet"], index=1)
fname = st.sidebar.text_input("File name (without extension)", value="filtered_dataset")
if fmt == "CSV":
    csv = df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Download", data=csv, file_name=f"{fname}.csv", mime="text/csv")
else:
    try:
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq
        buf = io.BytesIO()
        table = pa.Table.from_pandas(df)
        pq.write_table(table, buf, compression="snappy")
        st.sidebar.download_button("Download", data=buf.getvalue(), file_name=f"{fname}.parquet", mime="application/octet-stream")
    except Exception as e:
        st.sidebar.error(f"Parquet export needs pyarrow: {e}")
