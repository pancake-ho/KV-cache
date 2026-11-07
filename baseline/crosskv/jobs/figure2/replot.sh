#!/usr/bin/env bash
# No model inference or fitting. Run inside an allocated compute session.

set -euo pipefail
export FIG2_RUN_DIR="${1:?Usage: replot.sh /absolute/run/directory}"
REPO_ROOT="/data/surt321/repos/lab/kv_cache"
FIGURE2_DIR="${REPO_ROOT}/baseline/crosskv/jobs/figure2"
source "${FIGURE2_DIR}/lib.sh"
python -m xmodel_kv.figure2.plot --run-dir "${FIG2_RUN_DIR}"
