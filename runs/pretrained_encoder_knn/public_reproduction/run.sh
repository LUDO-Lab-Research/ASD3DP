#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_ROOT="$(cd "${RUN_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RELEASE_ROOT="${RELEASE_ROOT:?Set RELEASE_ROOT to the extracted ASD3DP release root}"
SPLIT_ROOT="${RUNS_ROOT}/splits/dcase2026_seed13711"
STAGING_ROOT="${RUN_DIR}/work/staging/ASD3DP"

cd "${RUN_DIR}"
"${PYTHON_BIN}" scripts/download_sslam.py --verify-only
"${PYTHON_BIN}" scripts/stage_asd3dp_pairs.py \
  --train-manifest "${SPLIT_ROOT}/run20_train.csv" \
  --test-manifest "${SPLIT_ROOT}/run20_test.csv" \
  --release-root "${RELEASE_ROOT}" \
  --output-root "${STAGING_ROOT}" \
  --near-channel ch1
DCASE2026_CONFIG="${RUN_DIR}/config/data_config_asd3dp.yaml" \
  "${PYTHON_BIN}" run_residual_view.py --config config/run21_asd3dp.yaml
