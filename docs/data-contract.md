## Raw Dataset Contract (`datasets/original_data.csv`)

| Column Name          | Type / Format         | Units / Semantics                                 | Validation Rules & Notes                                              |
|----------------------|-----------------------|---------------------------------------------------|------------------------------------------------------------------------|
| `Ref`                | integer               | Internal record identifier                        | Optional; not used directly in modelling.                              |
| `Row`                | integer               | Row counter per CSV extract                        | Optional; may be duplicated.                                          |
| `NMI_UI`             | string                | National Meter Identifier                          | Optional categorical; may be null/blank.                               |
| `METER_UI`           | string                | Unique meter identifier used as series ID          | Required; duplicates with identical timestamp are deduplicated.        |
| `AGGREGATE_DATE`     | ISO 8601 datetime w/ timezone (UTC+08:00) | Aggregated interval start                          | Required; converted to timezone-naive UTC+8 during ingest.             |
| `Date`               | string (`DD/MM/YYYY`) | Human-readable date                                | Optional convenience column; not used in modelling.                    |
| `Time`               | string (`H:MM`)       | Hour within day                                    | Optional; not used in modelling.                                       |
| `AGGREGATE_YEAR`     | integer               | Year component                                     | Should match `AGGREGATE_DATE`.                                        |
| `AGGREGATE_MONTH`    | integer (1–12)        | Month component                                    | Should match `AGGREGATE_DATE`.                                        |
| `AGGREGATE_DAY`      | integer (1–31)        | Day-of-month component                             | Should match `AGGREGATE_DATE`.                                        |
| `Error Check day`    | integer / flag        | Quality marker                                     | Optional quality signal; retained as numeric feature.                  |
| `AGGREGATE_HOUR`     | integer (0–23)        | Hour component                                     | Should match `AGGREGATE_DATE`.                                        |
| `Error Check Hour`   | integer / flag        | Quality marker                                     | Optional quality signal; retained as numeric feature.                  |
| `DELIVERED_VALUE`    | float                 | Energy delivered in kWh for the interval           | **Target variable**; must be non-negative.                             |
| `Daily Energy Usage` | float                 | Daily energy aggregation                           | Optional; retained as numeric feature.                                 |
| `RECEIVED_VALUE`     | float                 | Energy received/back-feed in kWh                   | Optional; retained as numeric feature.                                 |
| `Quarter`            | integer (1–4)         | Calendar quarter                                   | Optional categorical (treated as numeric).                             |
| `power_zero`         | integer (0/1)         | Flag indicating zero power                         | Optional; converted to numeric feature.                                |
| `daily_energy_zero`  | integer (0/1)         | Flag indicating zero daily energy                  | Optional; converted to numeric feature.                                |

### Required Columns

The training pipeline requires at minimum:

- `METER_UI` (series identifier)
- `AGGREGATE_DATE` (timestamp)
- `DELIVERED_VALUE` (target)

Other numeric columns are kept as candidate features. Non-numeric columns (`Date`, `Time`, `NMI_UI`) are dropped prior to model fitting.

### Sampling & Frequency

- Expected frequency: hourly (`freq: "H"`). Missing hours will reduce training data after feature engineering; fill or interpolate prior to ingestion if continuity is required.
- Duplicate rows with the same `(METER_UI, AGGREGATE_DATE)` are removed, keeping the first occurrence.

### Data Quality Rules

- Timestamps must be monotonically sortable within each meter series.
- `DELIVERED_VALUE` should be finite and non-null; rows with missing targets will be dropped during feature engineering.
- Rolling features require sufficient history (equal to the largest rolling window). Ensure data windows cover at least the maximum lag/rolling periods configured in `configs/experiment.yaml`.

By adhering to this contract, the training and prediction pipelines can operate without additional validation scaffolding.
