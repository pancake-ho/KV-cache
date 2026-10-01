#!/usr/bin/env bash

set -euo pipefail
REPO_ROOT="/data/surt321/repos/lab/kv_cache"
FIGURE2_DIR="${REPO_ROOT}/baseline/crosskv/jobs/figure2"
source "${FIGURE2_DIR}/lib.sh"

: "${FIG2_RUN_DIR:?Set FIG2_RUN_DIR}"
fig2_require_gpu
nvidia-smi > "${FIG2_RUN_DIR}/provenance/nvidia_smi_probe.txt"
python -m xmodel_kv.figure2.probe --run-dir "${FIG2_RUN_DIR}" --device cuda:0
python -m xmodel_kv.figure2.plot --run-dir "${FIG2_RUN_DIR}"
python - "${FIG2_RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
print(json.dumps(json.loads((root / 'results/summary.json').read_text()), indent=2))
print('Final plots:', root / 'results')
PY
