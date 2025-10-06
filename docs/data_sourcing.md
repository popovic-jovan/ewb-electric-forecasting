# Data Sourcing and Layout

This project separates code from data. All generated data and large files live under `data/` and are gitignored.

## Folders

- `data/raw/`          — original inputs (as provided by data owners)
- `data/external/`     — any third‑party data you did not generate
- `data/interim/`      — merged/cleaned intermediate outputs (e.g., join of weather + electricity)
- `data/processed/`    — model‑ready feature tables (e.g., dataset_ml.csv)

Use `.gitkeep` placeholders so folders exist in the repo without containing data.

## Placing Source Files

Place your source data files here:

- Weather CSVs/XLSX: put into `data/raw/weather/` (recommend creating this subfolder)
- Electricity export (hourly): `data/raw/electricity/`

Update the script configs to point to these locations or adjust the file lists/globs.

## Generated Artifacts

- `dataset_engineering/merge_weather_energy.py` writes merged outputs to `data/interim/`:
  - `merged_electricity_weather.csv`
  - `merged_electricity_weather.parquet`

- `dataset_engineering/feature_engineering.py` reads the merged CSV from `data/interim/` and writes model datasets to `data/processed/`:
  - `dataset_timeseries.csv`
  - `dataset_ml.csv`
  - `dataset_dl.csv`

These locations are ignored by git (see `.gitignore`).

### Model Training Outputs

- Scripts under `modelling/` now default to writing into `models/<model_name>/<run_timestamp>/`.
- Supply `--run_name` (and optionally `--no_timestamp`) to control the run subdirectory when invoking a trainer, e.g. `python modelling/prophet.py --run_name baseline`.

## Migrating Existing Files

If the repository already contains generated CSVs, remove them from git history and move them into the new layout:

1. Move files locally (examples):
   - `merged_electricity_weather.csv` → `data/interim/`
   - `model_datasets/dataset_*.csv` → `data/processed/`
   - `outputs_*` (model outputs) → keep or move under `models/` or `data/processed/` as appropriate

2. Stop tracking large artifacts already committed:
   - `git rm --cached -r model_datasets outputs_* merged_electricity_weather.csv`
   - `git commit -m "stop tracking generated artifacts; use data/ layout"`

3. Optionally rewrite history with `git filter-repo` or `git lfs migrate` if very large files bloat the repo.

## Reproducibility Tips

- Keep configs (paths, thresholds) in code or YAML and avoid absolute user‑specific paths.
- Use `requirements.txt` or `pyproject.toml` and pin versions.
- Consider DVC or lakeFS to version large data separately from git.
