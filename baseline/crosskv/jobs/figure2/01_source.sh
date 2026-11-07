#!/usr/bin/env bash

set -euo pipefail
REPO_ROOT="/data/surt321/repos/lab/kv_cache"
FIGURE2_DIR="${REPO_ROOT}/baseline/crosskv/jobs/figure2"
source "${FIGURE2_DIR}/lib.sh"

: "${FIG2_DATASET:?Export FIG2_DATASET to the local FineWeb-Edu file}"
: "${FIG2_RUN_DIR:?Set FIG2_RUN_DIR}"
mkdir -p "${FIG2_RUN_DIR}/provenance"
# Record installed versions with stdlib metadata. Runtime does not need pip.
fig2_check_env --output "${FIG2_RUN_DIR}/provenance/environment_source.json"
fig2_require_gpu
nvidia-smi > "${FIG2_RUN_DIR}/provenance/nvidia_smi_source.txt"
python -m compileall -q src/xmodel_kv/figure2
python -m pytest -q tests/test_figure2.py tests/test_figure2_tiny_qwen.py

smoke_args=()
if [[ "${FIG2_MODE:-full}" == smoke ]]; then
    smoke_args+=(--smoke)
fi
python -m xmodel_kv.figure2.prepare \
    --config "${FIG2_ROOT}/configs/figure2_qwen3_1p7b_4b.json" \
    --dataset "${FIG2_DATASET}" --run-dir "${FIG2_RUN_DIR}" "${smoke_args[@]}"
python -m xmodel_kv.figure2.extract --run-dir "${FIG2_RUN_DIR}" --role source
