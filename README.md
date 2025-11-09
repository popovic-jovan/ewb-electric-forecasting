# honours_project

Unified experimentation workspace for energy forecasting models.

## Quick Start

Train a model (default outputs go to `models/<name>/`):

```bash
python apps/cli.py train --model xgboost
```

Speed up iteration with a reduced configuration:

```bash
python apps/cli.py train --model sarimax --quick
```

Inspect available models:

```bash
python apps/cli.py list-models
```

Hyper-parameter tuning:

```bash
python apps/cli.py tune --model sarima --tune-config configs/model/sarima.yaml
```

Generate forecasts from a saved checkpoint:

```bash
python apps/cli.py predict \
  --model prophet \
  --checkpoint models/prophet/artifacts/prophet_model.json \
  --data path/to/recent_observations.csv \
  --output forecasts.csv
```

## Streamlit Dashboard

Launch the interactive UI to review metrics and produce live forecasts:

```bash
streamlit run apps/ui/app.py
```

The dashboard looks for trained checkpoints under `models/<model>/artifacts`, displays the latest reports, and lets you upload fresh CSV data for on-demand predictions.

## Project Structure (selected)

```
apps/
  cli.py        # unified CLI for train/tune/predict
  ui/app.py     # Streamlit dashboard
configs/
  model/        # per-model configuration files
models/         # generated artifacts, metrics, tuning studies
src/core/
  data/         # aggregation utilities
  models/       # model implementations (xgboost, sarimax, sarima, prophet, lstm)
  pipelines/    # orchestration layers for train/tune/predict
```

## Notes

- All models operate on the aggregated load series defined in `configs/dataset.yaml`.
- `--quick` trims search grids and training history to drastically cut runtime during development.
- Metric reports, predictions, and tuning summaries are written alongside each model inside `models/<name>/`.
