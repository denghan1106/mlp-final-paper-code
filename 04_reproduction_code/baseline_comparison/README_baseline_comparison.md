# Baseline localization tests

## 20260430 true-inputs baselines

For the `20260430_true_inputs` analysis, use:

```bash
python3 mlp/baseline/run_20260430_true_inputs_baselines.py
```

This script reads the three fixed validation splits from
`mlp/20260430_mlp_analysis`:

- `validation_split1`: `results_162_w120_blend30_anchor_scale_seed1_good`
- `validation_split7`: `results_162_w120_blend30_anchor_auto5_seed7_good`
- `validation_split10`: `results_162_w120_blend30_anchor_auto5_seed7`

For each split, it keeps the exact same train/test points as the improved MLP
run and writes results under `mlp/baseline/20260430_true_inputs_results`.

Experiments:

- `sensor1_current_improved_mlp`: current improved MLP recipe, restricted to
  the right-side four-channel sensor (`right_input23`, `right_input24`,
  `right_input25`, `right_input26`).
- `plain_mlp`: ordinary joint MLP, no separate x model, no x stretch, no anchor
  correction, no edge weighting.
- `grnn`: GRNN baseline on point-mean features.
- `extra_trees`: ExtraTrees baseline on point-mean features.

The script writes `train_point_predictions.csv`, `test_point_predictions.csv`,
`metrics.json`, and a top-level `summary_metrics.csv`.

## Cross-day baselines

This folder contains baseline runs separated from the current improved MLP
experiments.

Default command:

```bash
python3 mlp/baseline/run_baselines.py
```

Default validations:

- `validation_20260429`: train on `20260430_true_inputs` plus the 29 target
  calibration records, test on the remaining 34 records from `20260429`.
- `validation_20260428`: same split pattern, target dataset `20260428`.
- `validation_20260429_2`: train on `20260430_true_inputs` plus the 35 target
  calibration records used in the y-anchor run, test on the remaining 28
  records from `20260429_2`; target feature extraction uses `20260429_2:12`.

Default experiments:

- `sensor1_current_mlp_ensemble`: current MLP recipe with only the four-channel
  right-side sensor (`right_input23`, `right_input24`, `right_input25`,
  `right_input26`).
- `full_plain_mlp`: ordinary single MLP regressor with full features.
- `full_grnn`: GRNN baseline on per-point mean features.
- `full_extra_trees`: ExtraTrees baseline on per-point mean features.

Each result directory writes:

- `train_point_predictions.csv`
- `test_point_predictions.csv`
- dataset-specific train/test prediction CSVs
- `metrics.json`

The top-level output folder also writes `summary_metrics.csv`.
