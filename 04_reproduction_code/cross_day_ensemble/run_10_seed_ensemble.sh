#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PY="${PYTHON:-python3}"
SCRIPT="mlp/20260430true162_predict_20260429_anchor/train_20260430true162_predict_20260429.py"
ENSEMBLE="mlp/20260430true162_predict_20260429_anchor/ensemble_point_predictions.py"
OUT_ROOT="mlp/20260430true162_predict_20260429_anchor"
CAL_IDS=(1 2 3 4 5 6 7 8 9 10 18 19 27 28 32 36 37 45 46 54 55 56 57 58 59 60 61 62 63)

RESULT_DIRS=()
for SEED in 1 2 3 4 5 6 7 8 9 10; do
  OUT_DIR="${OUT_ROOT}/results_joint_record_median_calib29_w2_alpha0001_seed${SEED}"
  RESULT_DIRS+=("$OUT_DIR")
  if [[ -f "${OUT_DIR}/metrics.json" ]]; then
    echo "Skip existing ${OUT_DIR}"
    continue
  fi
  "$PY" "$SCRIPT" \
    --output-dir "$OUT_DIR" \
    --max-windows-per-point 120 \
    --signal-normalization record_median \
    --calibration-sample-weight 2 \
    --model-random-state "$SEED" \
    --alpha 0.0001 \
    --train-calibration-record-ids "${CAL_IDS[@]}"
done

"$PY" "$ENSEMBLE" \
  --output-dir "${OUT_ROOT}/results_joint_record_median_calib29_w2_alpha0001_seed1_10_ensemble" \
  --result-dir "${RESULT_DIRS[@]}"
