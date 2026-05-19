# 20260430 True Inputs 162 -> 20260429 63 Localization

This folder tests training on all 162 points in `mlp/20260430_true_inputs`
and predicting records `1..63` in `mlp/20260429`.  The 63 records have the
same coordinates as the first 63 points of the 18x9 20260430 grid.

The training wrapper reuses the sklearn MLP workflow from:

`mlp/20260428_20260429_mlp_localization/mlp_localization_20260428_20260429_sklearn_mlp_style.py`

Default 5 anchors are record ids `1, 9, 32, 55, 63`, corresponding to the
four corners and center of the 7x9 20260429 area.

## Run

```bash
bash mlp/20260430true162_predict_20260429_anchor/run_initial_tests.sh
```

Summarize existing results:

```bash
python3 mlp/20260430true162_predict_20260429_anchor/summarize_results.py
```

Generate 63-point-only visualizations:

```bash
python3 mlp/20260430true162_predict_20260429_anchor/visualize_63_results.py
```

Run the current 10-seed ensemble recipe on another 63-record dataset:

```bash
bash mlp/20260430true162_predict_20260429_anchor/run_10_seed_ensemble_for_dataset.sh \
  mlp/20260428 20260428

bash mlp/20260430true162_predict_20260429_anchor/run_10_seed_ensemble_for_dataset.sh \
  mlp/20260429_2 20260429_2 20260429_2:12
```

This script keeps only the final `seed1_10_ensemble` directory by default.
Set `KEEP_SEED_RESULTS=1` when individual seed folders are needed for
debugging.  For quick tests, limit seeds with `SEEDS="1 2"`.
Set `FUNDAMENTAL_FREQ=18` to run the same recipe with 18Hz feature extraction
instead of the default 15Hz.
For `20260428`, set `KEEP_20260428_INPUT29=1` to keep measured input29 instead
of the default theoretical replacement:

```bash
SEEDS="1 2" KEEP_20260428_INPUT29=1 \
  bash mlp/20260430true162_predict_20260429_anchor/run_10_seed_ensemble_for_dataset.sh \
  mlp/20260428 20260428_keep_input29
```
Additional calibration records can be appended with `EXTRA_CAL_IDS`, for
example:

```bash
SEEDS="1 2" EXTRA_CAL_IDS="34 40 41" \
  bash mlp/20260430true162_predict_20260429_anchor/run_10_seed_ensemble_for_dataset.sh \
  mlp/20260429_2 20260429_2_yanchors_34_40_41 20260429_2:12
```

Use `CAL_IDS_OVERRIDE` to replace the default calibration set, and `CAL_WEIGHT`
to change its sample-weight multiplier:

```bash
SEEDS="1 2" CAL_IDS_OVERRIDE="1 5 9 28 32 36 55 59 63" CAL_WEIGHT=5 \
  bash mlp/20260430true162_predict_20260429_anchor/run_10_seed_ensemble_for_dataset.sh \
  mlp/20260429_2 20260429_2_anchors9 20260429_2:12
```

Each training run now also writes its own 63-point plots directly into that
run's result folder:

`test_<dataset>_points_63.png`, `test_<dataset>_heatmap_63.png`, and
`comparison_metrics_63.csv`.  `test_point_predictions.png` and
`test_error_heatmap.png` are also overwritten with the 7x9 63-point view.

## Initial Results

| Experiment | Mean error | Median error | RMSE x | RMSE y | Max error |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline original style | 18.419 | 18.288 | 18.349 | 3.697 | 25.497 |
| anchor5 bias original style | 4.389 | 4.239 | 2.814 | 3.697 | 8.318 |
| anchor5 scale-bias original style | 4.360 | 4.273 | 2.758 | 3.697 | 8.246 |
| baseline record_median | 1.718 | 1.554 | 0.925 | 1.713 | 3.991 |
| anchor5 scale-bias record_median | 1.695 | 1.530 | 0.896 | 1.713 | 3.999 |
| 2D offset5 on baseline record_median, non-anchor | 1.622 | 1.484 | 0.885 | 1.633 | 4.001 |
| 2D affine9 on baseline record_median, non-anchor | 1.626 | 1.426 | 0.886 | 1.597 | 3.711 |
| joint MLP + record_median + 29 calibration records, predicts remaining 34 | 0.765 | 0.725 | 0.535 | 0.693 | 2.152 |
| window_mlp_classifier_blend weight 0.3 | 1.876 | 1.506 | 1.335 | 1.713 | 4.651 |
| window_mlp_classifier_blend weight 0.1 | 1.914 | 1.551 | 1.437 | 1.713 | 5.547 |
| separate window_mlp x only | 1.943 | 1.529 | 1.503 | 1.713 | 6.000 |
| previous xprob script, no 20260429 calibration | 2.161 | 1.989 | 1.576 | 2.082 | 7.843 |
| previous xprob script, 29 calibration records, predicts remaining 34 | 1.082 | 0.965 | 0.953 | 0.858 | 3.219 |
| joint MLP + record_median + 29 calibration + weight 2 | 0.753 | 0.730 | 0.496 | 0.678 | 1.925 |
| joint MLP + record_median + 29 calibration + weight 2 + seed 1 | 0.571 | 0.393 | 0.358 | 0.623 | 2.064 |
| joint MLP + record_median + 29 calibration + weight 2 + seed 1 + alpha 0.0001 | 0.568 | 0.431 | 0.390 | 0.591 | 2.091 |
| joint MLP + record_median + 29 calibration + weight 2 + seed 7 | 0.642 | 0.578 | 0.349 | 0.639 | 1.571 |
| joint MLP + record_median + 29 calibration + weight 2 + alpha 0.0001 + seeds 1-10 ensemble | 0.556 | 0.452 | 0.353 | 0.549 | 1.603 |
| same 20260430 -> 20260429 ensemble recipe, 18Hz features | 0.751 | 0.631 | 0.473 | 0.747 | 2.242 |
| same ensemble recipe, target 20260428 | 1.166 | 0.972 | 1.099 | 0.755 | 2.494 |
| same ensemble recipe, target 20260429_2 with 12Hz target features | 1.018 | 0.815 | 0.620 | 1.001 | 2.586 |
| 20260428, seeds 1-2, default theoretical input29 | 1.274 | 1.019 | 1.199 | 0.852 | 2.621 |
| 20260428, seeds 1-2, keep measured input29 | 1.584 | 1.370 | 1.553 | 0.991 | 4.028 |
| 20260429_2 12Hz, seeds 1-2, original calibration | 1.093 | 0.820 | 0.612 | 1.125 | 2.568 |
| 20260429_2 12Hz, seeds 1-2, + records 40,41 | 0.900 | 0.894 | 0.648 | 0.779 | 2.755 |
| 20260429_2 12Hz, seeds 1-2, + records 34,40 | 0.896 | 0.765 | 0.546 | 0.875 | 2.205 |
| 20260429_2 12Hz, seeds 1-2, + records 34,40,41 | 0.842 | 0.891 | 0.587 | 0.710 | 1.699 |
| 20260429_2 12Hz, seeds 1-2, + y=3/4 internal calibration | 0.626 | 0.678 | 0.412 | 0.573 | 1.323 |
| 20260429_2 12Hz, seeds 1-10, + y=3/4 internal calibration | 0.649 | 0.691 | 0.459 | 0.564 | 1.233 |
| 20260429_2 12Hz, seeds 1-2, x=4 column + y=4 row only, 15 calibration records | 1.999 | 1.605 | 1.634 | 1.693 | 4.790 |
| 20260429_2 12Hz, seeds 1-2, 29 baseline calibration + x=4/y=4 cross | 0.635 | 0.413 | 0.575 | 0.668 | 2.958 |
| 20260429_2 12Hz, seeds 1-2, full edge + records 40,41,42,49,50,51 | 0.875 | 0.784 | 0.559 | 0.878 | 3.092 |
| 20260429_2 12Hz, seeds 1-2, replace top-edge internal anchors with y=3/4 anchors | 0.700 | 0.662 | 0.526 | 0.568 | 1.436 |
| 20260429_2 12Hz, seeds 1-2, replace bottom-edge internal anchors with y=3/4 anchors | 1.038 | 0.814 | 1.093 | 0.850 | 4.621 |
| 20260429_2 12Hz, seeds 1-2, replace side-edge anchors with y=3/4 anchors | 0.984 | 0.791 | 0.744 | 0.861 | 2.654 |
| 20260429_2 12Hz, seeds 1-2, four corners only | 2.260 | 2.234 | 2.019 | 1.564 | 5.678 |
| 20260429_2 12Hz, seeds 1-2, four corners + center | 1.796 | 1.697 | 1.648 | 1.280 | 5.881 |
| 20260429_2 12Hz, seeds 1-2, four corners + records 34,40,41 | 1.754 | 1.237 | 1.690 | 1.329 | 5.675 |
| 20260429_2 12Hz, seeds 1-2, 9 spatial anchors, weight 2 | 1.142 | 0.890 | 1.140 | 0.862 | 4.779 |
| 20260429_2 12Hz, seeds 1-2, 9 spatial anchors, weight 5 | 1.098 | 0.926 | 1.101 | 0.855 | 4.745 |
| 20260429_2 12Hz, seeds 1-2, 9 spatial anchors + records 34,40,41, weight 5 | 1.024 | 0.903 | 0.961 | 0.889 | 4.557 |

The first useful improvement is `record_median` normalization.  The anchor-only
x correction fixes the large x offset in the unnormalized case, but y remains
poor; after `record_median`, anchors only add a small x improvement.

In the current cross-day 20260430 -> 20260429 prediction, the earlier
`window_mlp_classifier_blend` path is not helpful.  Both tested classifier
weights worsen x RMSE compared with the joint MLP baseline; the y result is
unchanged because the blend only replaces the x coordinate.

The retained `mlp/20260430_mlp_analysis` metrics are mostly same-day 20260430
random hold-out tests.  Those results reached roughly 0.66-0.80 mean error,
but that performance does not transfer directly to 20260429.  Re-running the
old full 20260430 -> 20260429 x-probability script without 20260429
calibration gives 2.16 mean error.  With 29 labeled 20260429 calibration
records added to training, the remaining 34 records improve to 1.08 mean
error, but that is a calibration-heavy setting rather than pure cross-day
prediction.

Using the same 29 calibration records with the simpler joint MLP and
`record_median` is better: the remaining 34 records reach 0.765 mean error.
Increasing the sample weight of the 20260429 calibration records to 2 and
tuning only the MLP random seed improves the remaining 34 records to about
0.57 mean error; seed 7 is not the best mean error, but gives the lowest
maximum error among the tested runs.
Taking the point-wise average of seeds 1-10 with `alpha=0.0001` improves mean
error slightly further to 0.556 and reduces the maximum error compared with the
best single mean-error model.

Switching both 20260430 and 20260429 feature extraction from 15Hz harmonics to
18Hz harmonics does not improve the cross-day prediction.  The 10-seed ensemble
with 18Hz gives `mean=0.751`, `rmse_x=0.473`, `rmse_y=0.747`, and `max=2.242`;
the main regression is y error.  On 20260430-only random hold-out, the same
18Hz idea is roughly neutral but still slightly worse than the 15Hz reference:
`mean=0.686` versus `0.669`.

Applying the same recipe to `20260428` and `20260429_2` is worse than the
original `20260429` result.  For `20260428`, the main residual is x direction
(`rmse_x=1.099`).  For `20260429_2`, the target feature extraction uses the
12Hz source frequency (`12/24/36/48Hz`, window size 298), but the main residual
is y direction (`rmse_y=1.001`).

Keeping measured `left_input29` for `20260428` does not improve the result.  In
a seeds 1-2 comparison, the default theoretical replacement gives
`mean=1.274`, `rmse_x=1.199`, while measured input29 gives `mean=1.584`,
`rmse_x=1.553`.  The measured-input29 residual is especially worse on row y=4.
Raw channel statistics also show that 20260428 measured input29 is less
consistent: the median correlation between measured input29 and the theoretical
mirror is about `0.245`, compared with about `0.569` for 20260429 and `0.525`
for 20260429_2.

The `20260429_2` y residual is systematic rather than random.  In the ensemble
result, the average y residual is `-0.689`, and rows with true y `3` and `4`
are predicted about `1.05` and `1.49` rows too low.  The target calibration
points themselves fit well in individual models (`rmse_y` around `0.14-0.17`),
so the issue is not failure to learn anchors.  The more likely cause is that
the 12Hz target data has a different spatial response between anchors: the
current border-plus-center calibration set constrains the edges and center, but
does not sufficiently constrain the interior of rows y=3 and y=4.

A single-seed control using 15Hz extraction for `20260429_2` was worse
(`mean=1.361`, `rmse_y=1.107`), so the 12Hz frequency setting is still the
better feature extraction choice.

Adding internal calibration points in the problem band improves `20260429_2`.
With only records `40,41`, the 2-seed `rmse_y` improves from `1.125` to
`0.779`; with records `34,40,41`, it improves to `0.710`.  The larger
six-record set `34,39,40,41,42,43` is still best in the 2-seed test
(`mean=0.626`, `rmse_y=0.573`).  The 10-seed ensemble lowers the maximum error
to `1.233`, but its mean error is slightly worse (`0.649`).  On the common
remaining 28 test points, the original 2-seed calibration gives `mean=0.827`,
`rmse_y=0.789`, while the six-record added-anchor model gives `mean=0.626`,
`rmse_y=0.573`.  The y bias on those common points is reduced from `-0.542` to
`-0.074`.

The top row calibration helps the boundary but does not fully control the
interior.  The original calibration includes the full bottom row y=0 and top
row y=6, plus sparse left/right edge points and one center point at record 32
(`x=4,y=3`).  In `20260429_2`, the y=5 test row is already reasonable, while
y=3/y=4 are pulled down.  This means the 12Hz spatial response is not behaving
like a simple linear interpolation from the top boundary to the center; the MLP
can fit the top-row anchors and still bend incorrectly in the rows below them.
Replacing six top-edge internal calibration records (`56,57,58,60,61,62`) with
the six y=3/4 internal records keeps the total calibration count at 29 and
still gives decent y error (`rmse_y=0.568`), but x drifts (`rmse_x=0.526`) and
the common 28-point mean error rises from `0.626` to `0.747`.  Releasing bottom
or side edge points is much worse.  The bottom edge is especially important for
x stability: replacing bottom-edge records `2,3,4,6,7,8` raises `rmse_x` to
`1.093` and max error to `4.621`.

The cross-shaped calibration set `x=4` column plus `y=4` row contains records
`5,14,23,32,37,38,39,40,41,42,43,44,45,50,59`.  Using only this 15-point
calibration set is poor (`mean=1.999`) because it lacks the corner and edge
frame needed to stabilize scale and x position.  Adding the missing cross
records on top of the 29-point baseline is useful on the remaining common test
points (`mean=0.534` on 23 points versus `0.618` for the 35-point y-anchor
2-seed run), but it no longer tests y=4 and has a larger maximum error
(`2.958`).

Using the full edge plus records `40,41,42,49,50,51` is weaker than the
six-record y-anchor set.  It controls the selected y=4/y=5 middle region
reasonably, but leaves rows y=2/y=3 biased downward (`err_y` around `-0.94`),
giving `mean=0.875` and `rmse_y=0.878`.

Four-corner-only calibration is not enough for `20260429_2`: with only records
`1,9,55,63`, the 2-seed mean error is `2.260`.  Adding only the center improves
to `1.796`, still poor.  A 9-point spatial layout (`1,5,9,28,32,36,55,59,63`)
is much better (`1.142`, or `1.098` with calibration weight 5), but still does
not match the 29-point baseline because x starts drifting.  A pure
post-calibration approach also failed: training with no target calibration and
then fitting a four-corner affine transform gives non-anchor mean error around
`3.01`, so the target points need to participate in model training.

For y-heavy residuals, run 2D post-calibration on the point predictions:

```bash
python3 mlp/20260430true162_predict_20260429_anchor/calibrate_predictions_2d.py \
  --source-dir mlp/20260430true162_predict_20260429_anchor/results_baseline_record_median \
  --method affine
```
