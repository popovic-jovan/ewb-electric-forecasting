# ============================
# SETUP
# ============================

# pip install pandas openpyxl pyarrow
import pandas as pd
import numpy as np
from pathlib import Path


# ============================
# 1) SMALL UTILITY FUNCTIONS
# ============================

def add_date_column(df, y='year', m='month', d='day', newcol='date'):
    """Build a single 'date' from separate year/month/day columns (keeps only the date)."""
    df = df.copy()
    df[y] = pd.to_numeric(df[y], errors='coerce')
    df[m] = pd.to_numeric(df[m], errors='coerce')
    df[d] = pd.to_numeric(df[d], errors='coerce')
    dt = pd.to_datetime(dict(year=df[y], month=df[m], day=df[d]), errors='coerce')
    df[newcol] = dt.dt.date
    return df


def load_daily_var(files, value_col, out_name):
    """
    Read multiple DAILY weather files (e.g., 2022 & 2023 for one variable) and stack them.
    Returns ['date', out_name] with ONE row per date.
    """
    dfs = []
    for f in files:
        if str(f).lower().endswith(('.xlsx', '.xls')):
            x = pd.read_excel(f)
        else:
            x = pd.read_csv(f)

        # >>> CHANGE_ME if your weather files use different column labels than 'year','month','day'
        x = add_date_column(x, 'year', 'month', 'day', 'date')

        if value_col not in x.columns:
            raise ValueError(
                f"{f} missing expected value column '{value_col}'. "
                f"Found: {x.columns.tolist()}\n"
                f">>> CHANGE_ME: set value_col for {out_name} to match your sheet header."
            )
        dfs.append(x[['date', value_col]].rename(columns={value_col: out_name}))

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    return out


def build_usage(usage_file, year_col='year', month_col='month', day_col='day', hour_col='hour',
                meter_col='meter_id', usage_col='current_usage'):
    """
    Load HOURLY usage and create 'timestamp' + 'date'.
    Returns ['meter_id','timestamp','date','current_usage','year','month','day','hour'].
    """
    if str(usage_file).lower().endswith(('.xlsx', '.xls')):
        u = pd.read_excel(usage_file)
    else:
        u = pd.read_csv(usage_file)

    # >>> CHANGE_ME here if your usage column names differ (or pass custom names when you call this)
    for c in [year_col, month_col, day_col, hour_col]:
        u[c] = pd.to_numeric(u[c], errors='coerce')

    ts = pd.to_datetime(dict(year=u[year_col], month=u[month_col], day=u[day_col]), errors='coerce') \
         + pd.to_timedelta(u[hour_col], unit='h')

    u['timestamp'] = ts.dt.tz_localize(None)   # WA: no DST; keep naive
    u['date'] = u['timestamp'].dt.date

    keep = [meter_col, 'timestamp', 'date', usage_col, year_col, month_col, day_col, hour_col]
    u = u[keep].rename(columns={
        meter_col: 'meter_id',
        usage_col: 'current_usage',
        year_col: 'year',
        month_col: 'month',
        day_col: 'day',
        hour_col: 'hour'
    })
    u = u.dropna(subset=['timestamp', 'meter_id'])
    return u


def merge_daily_weather(tmax_df, tmin_df, rain_df, solar_df):
    """Outer-join four daily variables on 'date' into one table."""
    weather = (tmax_df.merge(tmin_df, on='date', how='outer')
                        .merge(rain_df, on='date', how='outer')
                        .merge(solar_df, on='date', how='outer'))
    weather = weather.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    return weather


def engineer_features(df):
    """
    Add calendar, weather (daily), and usage-derived (lags/rolls) features.
    Expects hourly rows after weather + holiday merge.
    """
    df = df.sort_values(['meter_id', 'timestamp'])
    t = pd.to_datetime(df['timestamp'])

    # Calendar
    df['hour'] = t.dt.hour
    df['dow'] = t.dt.dayofweek
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['month'] = t.dt.month
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

    # Weather engineering (daily values repeated across 24 hours — intended)
    if {'tmax', 'tmin'}.issubset(df.columns):
        df['tmean'] = (df['tmax'] + df['tmin']) / 2
        df['CDD'] = (df['tmean'] - 24).clip(lower=0)
        df['HDD'] = (18 - df['tmean']).clip(lower=0)
    if 'rain' in df.columns:
        df['rain_flag'] = (df['rain'] > 0).astype(int)

    # Usage lags/rolls per meter
    grp = df.groupby('meter_id')['current_usage']
    df['usage_lag_1h']  = grp.shift(1)
    df['usage_lag_24h'] = grp.shift(24)
    df['usage_roll6h']  = grp.rolling(6,  min_periods=1).mean().reset_index(level=0, drop=True)
    df['usage_roll24h'] = grp.rolling(24, min_periods=1).mean().reset_index(level=0, drop=True)
    df['usage_vol24h']  = grp.rolling(24, min_periods=2).std().reset_index(level=0, drop=True)
    return df


# ============================
# 2) CONFIG: YOUR FILES
# ============================

# >>> CHANGE_ME: folder with your files
DATA_DIR = Path("/path/to/your/files")

# >>> CHANGE_ME: weather files (adjust filenames)
tmax_files  = [DATA_DIR/"tmax_2022.xlsx",  DATA_DIR/"tmax_2023.xlsx"]
tmin_files  = [DATA_DIR/"tmin_2022.xlsx",  DATA_DIR/"tmin_2023.xlsx"]
rain_files  = [DATA_DIR/"rain_2022.xlsx",  DATA_DIR/"rain_2023.xlsx"]
solar_files = [DATA_DIR/"solar_2022.xlsx", DATA_DIR/"solar_2023.xlsx"]

# >>> CHANGE_ME: exact column name holding the numeric value in each weather file
tmax_value_col  = "value"     # e.g., "TMax_C"
tmin_value_col  = "value"     # e.g., "TMin_C"
rain_value_col  = "value"     # e.g., "Rain_mm"
solar_value_col = "value"     # e.g., "Solar_MJm2"

# >>> CHANGE_ME: hourly usage file
usage_file = DATA_DIR/"usage_hourly.xlsx"

# >>> CHANGE_ME if your usage column names differ
usage_year_col  = 'year'
usage_month_col = 'month'
usage_day_col   = 'day'
usage_hour_col  = 'hour'
usage_meter_col = 'meter_id'        # e.g., "NMI"
usage_value_col = 'current_usage'   # e.g., "kWh"

# >>> CHANGE_ME: WA public holidays file (Date column like 20220103)
holidays_file = DATA_DIR/"wa_holidays.xlsx"  # or .csv


# ============================
# 3) LOAD WEATHER (DAILY)
# ============================

tmax  = load_daily_var(tmax_files,  value_col=tmax_value_col,  out_name='tmax')
tmin  = load_daily_var(tmin_files,  value_col=tmin_value_col,  out_name='tmin')
rain  = load_daily_var(rain_files,  value_col=rain_value_col,  out_name='rain')
solar = load_daily_var(solar_files, value_col=solar_value_col, out_name='solar')

weather = merge_daily_weather(tmax, tmin, rain, solar)
print("Daily weather rows:", len(weather), "| columns:", weather.columns.tolist())


# ============================
# 4) LOAD USAGE (HOURLY)
# ============================

usage = build_usage(
    usage_file,
    year_col=usage_year_col,
    month_col=usage_month_col,
    day_col=usage_day_col,
    hour_col=usage_hour_col,
    meter_col=usage_meter_col,
    usage_col=usage_value_col
)
print("Hourly usage rows:", len(usage), "| columns:", usage.columns.tolist())


# ============================
# 5) LOAD + PROCESS HOLIDAYS
# ============================

# The 'Date' column looks like 20220103 (YYYYMMDD). Convert it.
if str(holidays_file).lower().endswith(('.xlsx', '.xls')):
    hol = pd.read_excel(holidays_file)
else:
    hol = pd.read_csv(holidays_file)

# >>> CHANGE_ME if the holiday date column name isn't literally 'Date'
if 'Date' not in hol.columns:
    raise ValueError(f"Holidays file missing 'Date' column. Found: {hol.columns.tolist()}")

hol['Date'] = pd.to_datetime(hol['Date'], format='%Y%m%d', errors='coerce')
hol['date'] = hol['Date'].dt.date
hol['is_holiday'] = 1
hol = hol[['date', 'is_holiday']].drop_duplicates()
print("Holiday rows:", len(hol))


# ============================
# 6) MERGE (DAILY WEATHER + HOLIDAYS) INTO HOURLY USAGE
# ============================

# Weather: repeat daily values across all 24 hours (merge on 'date')
df = usage.merge(weather, on='date', how='left')

# Holidays: simple left merge on 'date' (fill non-holidays with 0)
df = df.merge(hol, on='date', how='left').fillna({'is_holiday': 0})

print("Merged rows:", len(df))


# ============================
# 7) FEATURE ENGINEERING
# ============================

df = engineer_features(df)


# ============================
# 8) SAVE RESULT
# ============================

out_path = DATA_DIR/"ml_ready_usage_weather.parquet"
df.to_parquet(out_path, index=False)
print("Saved:", out_path)

