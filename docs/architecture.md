## System Overview

The honours-project repository implements an hourly electricity demand forecasting workflow that is driven by YAML configuration files and modular pipeline components. The core flow is:

1. **Configuration ingestion** – dataset, experiment, and model YAML files define file paths, feature engineering options, training window cut-offs, and XGBoost hyper-parameters.
2. **Data loading & normalisation** – `src/core/data/preperation.py` reads the raw CSV, casts timestamps, aggregates per config, and removes duplicates.
3. **Feature engineering** – `src/core/features/time_series.py` derives calendar attributes, lagged targets, and rolling statistics before dropping rows with induced gaps.
4. **Temporal splitting** – `src/core/data/preperation.py` slices the frame into train/validation/test segments based on configured cut-off timestamps.
5. **Model training** – `src/core/models/xgboost.py` wraps `xgboost.XGBRegressor` to fit with early stopping and persist the trained booster.
6. **Metrics & artifact logging** – `src/core/evaluation/metrics.py` supplies regression metrics; the XGBoost model optionally logs results to MLflow and saves the trained model plus feature schema.
7. **Prediction & UI** – `predict_pipeline.py` reconstructs features and aligns them against the saved schema; `apps/ui/app.py` exposes a Streamlit interface for uploading data and visualising forecasts.

### High-Level Flow

```
        +-----------------+
        |  YAML Configs   |
        | (dataset/model/ |
        |   experiment)   |
        +--------+--------+
                 |
                 v
        +--------+--------+
        |  Data Loader    |
        | (CSV ingest &   |
        |  dedupe/sort)   |
        +--------+--------+
                 |
                 v
        +--------+--------+
        | Feature Builder |
        | (calendar, lag, |
        |  rolling stats) |
        +--------+--------+
                 |
                 v
        +--------+--------+
        | Time Splitter   |
        | (train/val/test)|
        +--------+--------+
                 |
                 v
        +--------+--------+
        | XGBoost Wrapper |
        | (fit/predict,   |
        |  save model)    |
        +--------+--------+
                 |
        +--------+--------+
        | Metrics &       |
        | MLflow Logging  |
        +--------+--------+
                 |
                 v
        +--------+--------+       +---------------------+
        | Saved Model &   +------>| Prediction Pipeline |
        | Feature Schema  |       | (feature rebuild &  |
        +--------+--------+       |  aligned inference) |
                 |                +----------+----------+
                 |                           |
                 |                           v
                 |                +----------+----------+
                 +----------------> Streamlit UI        |
                                  | (CSV upload + chart)|
                                  +---------------------+
```

## Key Modules

- **`src/core/data`** – ingestion and aggregation helpers (`preperation.py`).
- **`src/core/features`** – calendar/lag/rolling feature engineering.
- **`src/core/models`** – registered model wrappers with shared persistence helpers.
- **`src/core/pipelines`** – CLI-facing orchestration for training (`train.py`), prediction (`predict.py`), and hyper-parameter tuning (`tune.py`).
- **`src/core/io`** – configuration loader utilities.
- **`apps/train/train.py` & `apps/ui/app.py`** – CLI entry point for training/tuning, and Streamlit demo for ad-hoc inference.

## Configuration Contracts

- `configs/dataset.yaml` – file path, column names, sampling frequency, and time splits.
- `configs/experiment.yaml` – feature recipe, random seed, artifact locations, optional MLflow settings.
- `configs/model/xgboost.yaml` – base hyper-parameters, fit arguments, and optional tuning search space.

Together these config files keep experimentation reproducible and make it easy to evolve inputs, features, and training behaviour without modifying code.
