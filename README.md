# Final paper MLP code and results

This folder is reorganized by function. It contains only the final code and result files used for the paper.

## Folder layout

- `01_paper_summary_tables/`
  - paper-ready summary tables and result manifest.
  - start here for manuscript tables.

- `02_final_cross_day_mlp_result/`
  - final calibrated 10-seed MLP ensemble result for the 20260430 to 20260429 cross-day test.
  - includes metrics, prediction CSV files, and final plots.

- `03_detailed_baseline_results/`
  - detailed per-model result folders behind the paper summary tables.
  - `01_single_vs_dual_sensor_improved_mlp/`: single-sensor vs dual-sensor improved MLP.
  - `02_same_split_dual_sensor_model_comparison/`: same-split MLP / GRNN / tree baseline comparisons.
  - `03_cross_day_20260430_to_20260429_model_comparison/`: cross-day model comparison.

- `04_reproduction_code/`
  - scripts used to produce and assemble the final paper results.
  - `shared_mlp_core/`: shared sklearn MLP localization implementation.
  - `cross_day_ensemble/`: cross-day ensemble training, averaging, calibration, plotting, and summarizing scripts.
  - `baseline_comparison/`: baseline comparison and final table assembly scripts.

- `05_162_test_prediction_figures/`
  - three 162-point same-split test prediction figures used by the final validation references.
  - files are named by validation split: `validation_inner6`, `validation_split1`, and `validation_split7`.

## Excluded

Intermediate experiments, raw data folders, `__pycache__`, `.DS_Store`, Matplotlib cache folders, trained `model.joblib` files, and large `window_features.csv` feature-cache files are excluded from this final package.
