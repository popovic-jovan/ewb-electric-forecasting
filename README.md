# honours_project

Unified experimentation workspace for energy forecasting models.

## Data Requirements

- Always work off the **`source_table` sheet** from the Mowanjum dataset. This sheet already contains the aligned load and weather features expected by the pipelines.
- Point `DATA_PATH` (in the notebooks or CLI) to either the raw Excel sheet exported as CSV/Parquet or to a derivative file that preserves the same schema.

## Training & Running Models

The unified CLI (`apps/cli.py`) controls training, evaluation, prediction, and tuning for every model.

### Train a single model

```bash
python apps/cli.py train --model <model-name>
```

Available model slugs: `xgboost`, `sarimax`, `sarima`, `prophet`, `lstm`, plus any experimental entries defined under `configs/model/`.

Add `--quick` to reduce history length and grid sizes for rapid iteration.

### Train every model

```bash
for model in xgboost sarimax sarima prophet lstm; do
  python apps/cli.py train --model "$model"
done
```

All outputs (checkpoints, metrics, tuning traces) land in `models/<model>/`.

### Generate forecasts

```bash
python apps/cli.py predict \
  --model xgboost \
  --checkpoint models/xgboost/artifacts/best.ckpt \
  --data data/latest_observations.csv \
  --output forecasts.csv
```

## Tuning & Grid Search

The CLI exposes a consistent interface for sweeps. Config files under `configs/model/` define default search spaces; supply overrides for custom grids.

```bash
python apps/cli.py tune \
  --model sarima \
  --tune-config configs/model/sarima.yaml \
  --max-trials 50
```

To run full grid or Bayesian searches for *all* models:

```bash
for model in xgboost sarimax sarima prophet lstm; do
  python apps/cli.py tune --model "$model" --max-trials 75
done
```

`--quick` can also be combined with `tune` to sanity-check the pipeline before launching long searches.

## Streamlit Apps

### analytics dashboard (`apps/ui/app.py`)

```bash
streamlit run apps/ui/app.py
```

Shows the latest experiment metrics, plots feature diagnostics, and lets you upload CSVs for ad-hoc predictions. The app auto-discovers checkpoints under `models/<model>/artifacts`.

### live forecasting (`apps/ui/live.py`)

```bash
streamlit run apps/ui/live.py -- \
  --model xgboost \
  --checkpoint models/xgboost/artifacts/best.ckpt
```

This Streamlit surface streams predictions from `src/core/serving/live.py`, letting you pick a trained model and visualize near real-time forecasts while uploading new readings. Use it during demos or for lightweight monitoring without spinning up the full analytics dashboard.

## Project Structure (selected)

```
apps/
  cli.py        # unified CLI for train/tune/predict
  ui/app.py     # analytics dashboard
  ui/live.py    # live forecasting Streamlit entrypoint
configs/
  model/        # per-model configuration & tuning grids
models/         # generated artifacts, metrics, tuning studies
src/core/
  data/         # aggregation utilities
  models/       # model implementations (xgboost, sarimax, sarima, prophet, lstm)
  pipelines/    # orchestration layers for train/tune/predict
  serving/      # live inference helpers (used by live.py)
```

## Notes

- All models expect the preprocessed `source_table` schema; avoid mixing multiple sheets during training.
- `--quick` trims search grids and training history to drastically cut runtime during development.
- Metric reports, predictions, and tuning summaries are written alongside each model inside `models/<name>/`.
