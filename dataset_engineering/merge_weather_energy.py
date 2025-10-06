from __future__ import annotations
import sys
import re
from pathlib import Path
from tkinter.font import names
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# CONFIG
WEATHER_FILES = [
    "/Users/jovanpopovic/Documents/Honours 2025/Weather Data/Derby Aero Rainfall 2022/IDCJAC0009_003032_2022_Data.csv",
    "/Users/jovanpopovic/Documents/Honours 2025/Weather Data/Derby Aero Rainfall 2023/IDCJAC0009_003032_2023_Data.csv",
    "/Users/jovanpopovic/Documents/Honours 2025/Weather Data/Mowanjum Max Temp 2022/IDCJAC0010_003032_2022_Data.csv",
    "/Users/jovanpopovic/Documents/Honours 2025/Weather Data/Mowanjum Max Temp 2023/IDCJAC0010_003032_2023_Data.csv",
    "/Users/jovanpopovic/Documents/Honours 2025/Weather Data/Mowanjum Min Temp 2022/IDCJAC0011_003032_2022_Data.csv",
    "/Users/jovanpopovic/Documents/Honours 2025/Weather Data/Mowanjum Min Temp 2023/IDCJAC0011_003032_2023_Data.csv",
    "/Users/jovanpopovic/Documents/Honours 2025/Weather Data/Solar_Exposure 2022/IDCJAC0016_003084_2022_Data.csv",
    "/Users/jovanpopovic/Documents/Honours 2025/Weather Data/Solar_Exposure 2023/Solar_Exposure_2023.csv",
]

ELECTRICITY_FILE = "/Users/jovanpopovic/Documents/Honours 2025/Prepaid Mowanjum Meter Data_rec.csv"  # e.g. r"C:\data\energy\electricity_Apr2022_Apr2023.xlsx"
WEATHER_SHEETS: Dict[str, str] = {}
ELECTRICITY_SHEET = "Source Table"
WEATHER_LABEL_HINTS: Dict[str, str] = {}
HOUSEHOLD_ID_COL: str | None = None

# Output
OUTPUT_DIR = Path("./_merged_output")
OUTPUT_CSV = OUTPUT_DIR / "merged_electricity_weather.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "merged_electricity_weather.parquet"


# Helpers
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
          .str.replace(r"_+", "_", regex=True)
          .str.strip("_")
    )
    return df

def _to_date_safe(y=None, m=None, d=None, date_str=None):
    """
    Canonical date parser:
    - Prefer constructing from separate numeric year/month/day when available.
    - Otherwise parse a date string. Tries strict yyyymmdd first, then ISO/loose.
    Returns datetime.date.
    """
    import pandas as pd
    if y is not None and m is not None and d is not None:
        yv = pd.to_numeric(y, errors="coerce")
        mv = pd.to_numeric(m, errors="coerce")
        dv = pd.to_numeric(d, errors="coerce")
        return pd.to_datetime(dict(year=yv, month=mv, day=dv), errors="coerce").dt.date
    s = pd.Series(date_str, dtype="object").astype(str).str.strip()
    s_digits = s.str.replace(r"[^0-9]", "", regex=True)
    out = pd.to_datetime(s_digits.where(s_digits.str.len()==8, None), format="%Y%m%d", errors="coerce")
    iso = pd.to_datetime(s.where(out.isna(), None), errors="coerce", utc=False)
    out = out.fillna(iso)
    return out.dt.date

def try_parse_single_date_col(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    df = df.copy()
    for candidate in ["date", "dt", "day_date"]:
        if candidate in df.columns:
            df["date"] = _to_date_safe(date_str=df[candidate])
            if df["date"].notna().any():
                return df, True
    return df, False

def parse_year_month_day(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    df = df.copy()
    cols = set(df.columns)
    def pick(names):
        for n in names:
            if n in cols:
                return n
        return None
    y = pick(["year","yr","yyyy","aggregate_year","agg_year"])
    m = pick(["month","mm","aggregate_month","agg_month"])
    d = pick(["day","date","dd","aggregate_day","agg_day"])
    if y and m and d:
        df["date"] = _to_date_safe(df[y], df[m], df[d])
        return df, True
    return df, False

def extract_hour(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Return df with an 'hour' int column (0..23) if possible."""
    df = df.copy()
    # From timestamp/datetime
    for cand in ["timestamp", "datetime", "date_time", "ts"]:
        if cand in df.columns:
            ts = pd.to_datetime(df[cand], errors="coerce")
            if ts.notna().any():
                df["hour"] = ts.dt.hour
                # Also set date if missing
                if "date" not in df.columns or df["date"].isna().all():
                    df["date"] = ts.dt.date
                return df, True
    # From time
    for cand in ["time", "hour"]:
        if cand in df.columns:
            # If 'hour' is already 0..23 integers
            if cand == "hour":
                hv = pd.to_numeric(df[cand], errors="coerce")
                if hv.notna().any():
                    df["hour"] = hv.astype("Int64")
                    return df, True
            # Else parse 'time' like '13:00'
            tv = pd.to_datetime(df[cand], errors="coerce")
            if tv.notna().any():
                df["hour"] = tv.dt.hour
                return df, True
    # Fallback: infer by row order within each date (assumes 24 consecutive rows per date)
    if "date" in df.columns:
        df = df.sort_values("date").copy()
        df["hour"] = df.groupby("date").cumcount() % 24
        return df, True
    return df, False

def detect_household_col(df: pd.DataFrame, override: str | None) -> str | None:
    if override and override in df.columns:
        return override
    candidates = [
        "household_id","household","house_id","house","meter_id","nmi","customer_id","account_id",
        "site_id","dwelling_id","property_id","participant_id","id","meter","home_id"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None  # okay if absent

def read_excel_any(path: Path, sheet: str | None) -> pd.DataFrame:
    if str(path).lower().endswith('.csv'):
        return pd.read_csv(path)
    if sheet is None:
        return pd.read_excel(path)
    return pd.read_excel(path, sheet_name=sheet)

def parse_weather_file(path: Path, sheet: str | None, label_hint: str | None = None) -> pd.DataFrame:
    df = read_excel_any(path, sheet)
    df = normalize_cols(df)

    # Create 'date'
    df, ok1 = try_parse_single_date_col(df)
    if not ok1:
        df, ok2 = parse_year_month_day(df)
        if not ok2:
            raise ValueError(f"Could not parse date columns in weather file: {path.name}")

    # Define columns to exclude (metadata + date columns)
    exclude_cols = {
        "date", "year", "yr", "yyyy", "month", "mm", "day", "dd",
        "aggregate_year", "aggregate_month", "aggregate_day",
        "agg_year", "agg_month", "agg_day",
        "quality", "bureau_of_meteorology_station_number", "product_code"  # Add metadata columns to exclude
    }

    # Identify measurement columns
    value_cols = [c for c in df.columns if c not in exclude_cols]

    if not value_cols:
        raise ValueError(f"No measurement columns found in weather file: {path.name}")

    # If there's exactly one value column, try to rename to a friendly label
    if len(value_cols) == 1:
        vc = value_cols[0]
        if label_hint:
            value_cols_final = [label_hint]
            df = df.rename(columns={vc: label_hint})
        else:
            # Try infer from header name
            value_cols_final = [vc]
    else:
        value_cols_final = value_cols  # keep all

    out = df[["date"] + value_cols_final].copy()
    # Aggregate in case file contains duplicates per date
    out = out.groupby("date", as_index=False).agg("first")
    return out

def parse_weather_files(paths: List[Path], sheet_map: Dict[str,str], label_hints: Dict[str,str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        sh = sheet_map.get(p.name)
        lh = label_hints.get(p.name)
        w = parse_weather_file(p, sh, lh)
        audit_weather_frame(w, p.name) # TEMP: remove after debugging
        frames.append(w)
    if not frames:
        raise ValueError("No weather files provided.")
    # Merge on date (outer, then consolidate)
    weather = frames[0]
    for w in frames[1:]:
        weather = weather.merge(w, on="date", how="outer")
    #Collapse x/y into one column
    weather = _coalesce_duplicate_measurements(weather)
    # Sort & de-dup
    weather = weather.sort_values("date").drop_duplicates("date", keep="last")
    return weather

def parse_electricity_file(path: Path, sheet: str | None, household_override: str | None) -> Tuple[pd.DataFrame, str | None]:
    df = read_excel_any(path, sheet)
    df = normalize_cols(df)

    # Build date
    df, ok1 = try_parse_single_date_col(df)
    if not ok1:
        df, ok2 = parse_year_month_day(df)
        if not ok2:
            raise ValueError(f"Could not parse date columns in electricity file: {path.name}")

    # Ensure date column is datetime.date type and remove rows with NaN dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    df = df.dropna(subset=['date'])  # Remove rows with NaN dates

    # Extract hour
    df, okh = extract_hour(df)
    if not okh:
        raise ValueError("Could not infer 'hour' in electricity file. Add an 'hour' or 'timestamp' column.")

    # Detect household id
    hh_col = detect_household_col(df, household_override)

    # Ensure hour is int 0..23
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
    # Basic sanity filter
    df = df[(df["hour"] >= 0) & (df["hour"] <= 23)].copy()

    return df, hh_col

def expand_weather_to_hourly(weather_daily: pd.DataFrame) -> pd.DataFrame:
    weather_daily = weather_daily.copy()
    weather_daily["date"] = pd.to_datetime(weather_daily["date"]).dt.date
    hours = pd.DataFrame({"hour": list(range(24))})
    # Cross join (pandas >= 1.2 supports how='cross')
    weather_hourly = weather_daily.merge(hours, how="cross")
    # Sort
    weather_hourly = weather_hourly.sort_values(["date","hour"]).reset_index(drop=True)
    return weather_hourly

def validate_join(elec: pd.DataFrame, weather_hourly: pd.DataFrame) -> Dict[str,int]:
    # Anti-join checks
    ej = elec.merge(weather_hourly[["date","hour"]].drop_duplicates(), on=["date","hour"], how="left", indicator=True)
    missing_weather = int((ej["_merge"] == "left_only").sum())

    wj = weather_hourly.merge(elec[["date","hour"]].drop_duplicates(), on=["date","hour"], how="left", indicator=True)
    missing_elec = int((wj["_merge"] == "left_only").sum())

    return {"elec_rows_without_weather_match": missing_weather,
            "weather_rows_without_elec_match": missing_elec}

def _coalesce_duplicate_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse columns that only differ by pandas merge suffixes (_x, _y, repeated merges).
    For each base (e.g., 'maximum_temperature_degree_c'), take first non-null across
    duplicates, write into the base name, and drop the suffixed columns.
    """
    import re
    df = df.copy()
    groups = {}
    for c in df.columns:
        if c == "date":
            continue
        base = re.sub(r'(_x|_y)+$', '', c)
        groups.setdefault(base, []).append(c)

    for base, cols in groups.items():
        if len(cols) <= 1:
            continue
        df[base] = df[cols].bfill(axis=1).iloc[:, 0]
        drop_cols = [c for c in cols if c != base]
        if drop_cols:
            df = df.drop(columns=drop_cols)
    return df

def audit_weather_frame(df, name):
    s = pd.Series(pd.to_datetime(df["date"], errors="coerce").dt.date)
    print(f"[AUDIT] {name}: rows={len(df)}, unique_days={s.nunique()}, "
          f"min={s.min()}, max={s.max()}, NaT={int(s.isna().sum())}")
    # show any full-missing months
    if s.notna().any():
        vc = pd.Series([(d.year, d.month) for d in s.dropna()]).value_counts().sort_index()
        print(f"[AUDIT] {name} month counts:", dict(vc))


def main():
    # Resolve paths
    weather_paths = [Path(p) for p in WEATHER_FILES]
    elec_path = Path(ELECTRICITY_FILE) if ELECTRICITY_FILE else None

    if not weather_paths:
        raise SystemExit("Please list your 8 weather files in WEATHER_FILES.")

    if elec_path is None or not elec_path.exists():
        raise SystemExit("Please set ELECTRICITY_FILE to your hourly electricity Excel file.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Weather (daily)
    print("Reading weather files...")
    weather_daily = parse_weather_files(weather_paths, WEATHER_SHEETS, WEATHER_LABEL_HINTS)
    print(f"  Weather daily rows: {len(weather_daily):,}; columns: {list(weather_daily.columns)}")

    # 2) Electricity (hourly 49 households)
    print("Reading electricity file...")
    elec, hh_col = parse_electricity_file(elec_path, ELECTRICITY_SHEET, HOUSEHOLD_ID_COL)
    print(f"  Electricity rows: {len(elec):,}; columns: {list(elec.columns)}")
    if hh_col:
        print(f"  Detected household ID column: {hh_col}")
    else:
        print("  Household ID column not found; proceeding without it.")

    # Optional: restrict weather to electricity date range for faster join
    dmin, dmax = elec["date"].min(), elec["date"].max()
    weather_daily = weather_daily[(weather_daily["date"] >= dmin) & (weather_daily["date"] <= dmax)].copy()
    print(f"  Weather restricted to electricity date range: {dmin} → {dmax} ({len(weather_daily):,} days)")

    # Reindex weather to full daily calendar so missing days are explicit (NaN)
    full_days = pd.DataFrame({"date": pd.date_range(dmin, dmax, freq="D").date})
    weather_daily = full_days.merge(weather_daily, on="date", how="left")
    print("  Weather coverage: total days:", len(full_days),
        "   days with any weather:", int(weather_daily.drop(columns=["date"]).notna().any(axis=1).sum()))
    
    # 3) Expand weather to hourly
    weather_hourly = expand_weather_to_hourly(weather_daily)
    print(f"  Weather hourly rows: {len(weather_hourly):,}; columns: {list(weather_hourly.columns)}")

    # 4) Join on date + hour (left join keeps aligned rows)
    print("Joining electricity with hourly weather...")
    merged = elec.merge(weather_hourly, on=["date","hour"], how="left")
    print(f"  Merged rows: {len(merged):,}; columns: {list(merged.columns)}")

    keys_e = elec[["date","hour"]].drop_duplicates()
    keys_w = weather_hourly[["date","hour"]].drop_duplicates()
    miss = keys_e.merge(keys_w, on=["date","hour"], how="left", indicator=True)
    missing_cnt = int((miss["_merge"] == "left_only").sum())
    if missing_cnt:
        by_day = miss.loc[miss["_merge"] == "left_only", "date"].value_counts().sort_index()
        print("  Electricity date-hours without weather:", missing_cnt)
        print("  Sample missing-by-day:", dict(by_day.head(10)))
    else:
        print("  Electricity date-hours without weather: 0")
    
    # 5) Basic join validation
    stats = validate_join(elec, weather_hourly)
    print("  Join diagnostics:", stats)

    # 6) Save
    merged.to_csv(OUTPUT_CSV, index=False)
    try:
        merged.to_parquet(OUTPUT_PARQUET, index=False)
    except Exception as e:
        print(f"  Skipped Parquet write (install pyarrow or fastparquet to enable). Error: {e}")

    print(f"Done. Wrote:\n  - CSV: {OUTPUT_CSV}\n  - Parquet: {OUTPUT_PARQUET if OUTPUT_PARQUET.exists() else '(not written)'}")

if __name__ == "__main__":
    pd.set_option("future.no_silent_downcasting", True)
    main()
