#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_ROOT="$(cd "${RUN_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RELEASE_ROOT="${RELEASE_ROOT:?Set RELEASE_ROOT to the extracted ASD3DP release root}"
WORK_ROOT="${WORK_ROOT:-${RUN_DIR}/work}"
RAW_ROOT="${WORK_ROOT}/raw/3DPrinter"
CACHE_ROOT="${WORK_ROOT}/cache"
OUTPUT_ROOT="${WORK_ROOT}/outputs"
SPLIT_ROOT="${RUNS_ROOT}/splits/dcase2026_seed13711"
export WANDB_DIR="${WANDB_DIR:-${WORK_ROOT}/wandb}"
mkdir -p "${WANDB_DIR}"

"${PYTHON_BIN}" "${RUN_DIR}/scripts/stage_split.py" \
  --train-manifest "${SPLIT_ROOT}/run20_train.csv" \
  --test-manifest "${SPLIT_ROOT}/run20_test.csv" \
  --release-root "${RELEASE_ROOT}" \
  --output "${RAW_ROOT}"

for seed in 23711 23712 23713; do
  "${PYTHON_BIN}" "${RUN_DIR}/src/runner.py" \
    --raw-root "${RAW_ROOT}" \
    --cache-root "${CACHE_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/seed_${seed}" \
    --seed "${seed}" \
    --epochs 100 \
    --batch-size 1024 \
    --learning-rate 0.001 \
    --train-storage gpu \
    --wandb-project asd3dp_dcase_baseline_ae \
    --wandb-mode offline
done
