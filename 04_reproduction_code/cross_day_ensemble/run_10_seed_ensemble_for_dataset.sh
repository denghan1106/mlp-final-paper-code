#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <predict-data-dir> <output-tag> [dataset:freq ...]" >&2
  echo "Example: $0 mlp/20260429_2 20260429_2 20260429_2:12" >&2
  exit 2
fi

PREDICT_DIR="$1"
OUTPUT_TAG="$2"
shift 2
DATASET_FREQS=("$@")
if [[ -n "${SEEDS:-}" ]]; then
  read -r -a SEED_LIST <<< "$SEEDS"
else
  SEED_LIST=(1 2 3 4 5 6 7 8 9 10)
fi
if [[ ${#SEED_LIST[@]} -eq 0 ]]; then
  echo "No seeds provided." >&2
  exit 2
fi
LAST_SEED_INDEX=$((${#SEED_LIST[@]} - 1))
SEED_LABEL="seed${SEED_LIST[0]}_${SEED_LIST[$LAST_SEED_INDEX]}"

cd "$(dirname "$0")/../.."

PY="${PYTHON:-python3}"
SCRIPT="mlp/20260430true162_predict_20260429_anchor/train_20260430true162_predict_20260429.py"
ENSEMBLE="mlp/20260430true162_predict_20260429_anchor/ensemble_point_predictions.py"
OUT_ROOT="mlp/20260430true162_predict_20260429_anchor"
CAL_WEIGHT="${CAL_WEIGHT:-2}"
FUNDAMENTAL_FREQ="${FUNDAMENTAL_FREQ:-15}"
CAL_WEIGHT_LABEL="${CAL_WEIGHT//./p}"
CAL_IDS=(1 2 3 4 5 6 7 8 9 10 18 19 27 28 32 36 37 45 46 54 55 56 57 58 59 60 61 62 63)
if [[ -n "${CAL_IDS_OVERRIDE:-}" ]]; then
  read -r -a CAL_IDS <<< "$CAL_IDS_OVERRIDE"
fi
if [[ -n "${EXTRA_CAL_IDS:-}" ]]; then
  read -r -a EXTRA_CAL_ID_LIST <<< "$EXTRA_CAL_IDS"
  CAL_IDS+=("${EXTRA_CAL_ID_LIST[@]}")
fi

RESULT_DIRS=()
for SEED in "${SEED_LIST[@]}"; do
  OUT_DIR="${OUT_ROOT}/results_${OUTPUT_TAG}_joint_record_median_calib${#CAL_IDS[@]}_w${CAL_WEIGHT_LABEL}_alpha0001_seed${SEED}"
  RESULT_DIRS+=("$OUT_DIR")
  if [[ -f "${OUT_DIR}/metrics.json" ]]; then
    echo "Skip existing ${OUT_DIR}"
    continue
  fi
  ARGS=(
    "$SCRIPT"
    --predict-data-dir "$PREDICT_DIR"
    --output-dir "$OUT_DIR"
    --max-windows-per-point 120
    --signal-normalization record_median
    --calibration-sample-weight "$CAL_WEIGHT"
    --fundamental-freq "$FUNDAMENTAL_FREQ"
    --model-random-state "$SEED"
    --alpha 0.0001
    --train-calibration-record-ids "${CAL_IDS[@]}"
  )
  if [[ ${#DATASET_FREQS[@]} -gt 0 ]]; then
    ARGS+=(--dataset-fundamental-freqs "${DATASET_FREQS[@]}")
  fi
  if [[ "${KEEP_20260428_INPUT29:-0}" == "1" ]]; then
    ARGS+=(--keep-20260428-input29)
  fi
  "$PY" "${ARGS[@]}"
done

"$PY" "$ENSEMBLE" \
  --output-dir "${OUT_ROOT}/results_${OUTPUT_TAG}_joint_record_median_calib${#CAL_IDS[@]}_w${CAL_WEIGHT_LABEL}_alpha0001_${SEED_LABEL}_ensemble" \
  --result-dir "${RESULT_DIRS[@]}"

if [[ "${KEEP_SEED_RESULTS:-0}" != "1" ]]; then
  for OUT_DIR in "${RESULT_DIRS[@]}"; do
    rm -rf "$OUT_DIR"
  done
fi
