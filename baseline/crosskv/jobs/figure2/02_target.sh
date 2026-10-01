#!/usr/bin/env bash

set -euo pipefail
REPO_ROOT="/data/surt321/repos/lab/kv_cache"
FIGURE2_DIR="${REPO_ROOT}/baseline/crosskv/jobs/figure2"
source "${FIGURE2_DIR}/lib.sh"

: "${FIG2_RUN_DIR:?Set FIG2_RUN_DIR}"
fig2_require_gpu
nvidia-smi > "${FIG2_RUN_DIR}/provenance/nvidia_smi_target.txt"
python -m xmodel_kv.figure2.extract --run-dir "${FIG2_RUN_DIR}" --role target
