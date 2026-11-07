#!/usr/bin/env bash
# Shared environment. The default is the user's EXISTING lab environment.
set -euo pipefail
FIG2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export FIG2_CONDA_ROOT="${FIG2_CONDA_ROOT:-/data/${USER}/anaconda3}"
export FIG2_ENV="${FIG2_ENV:-${FIG2_CONDA_ROOT}/envs/lab}"
[[ -f "${FIG2_CONDA_ROOT}/etc/profile.d/conda.sh" ]] || { echo "Conda initialization missing: ${FIG2_CONDA_ROOT}" >&2; exit 2; }
[[ -x "${FIG2_ENV}/bin/python" ]] || { echo "Existing environment missing: ${FIG2_ENV}" >&2; exit 2; }
source "${FIG2_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${FIG2_ENV}"
cd "${FIG2_ROOT}"
export PYTHONPATH="${FIG2_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/data/${USER}/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONUNBUFFERED=1
if [[ -n "${FIG2_RUN_DIR:-}" ]]; then
    mkdir -p "${FIG2_RUN_DIR}/runtime_cache/matplotlib"
    export MPLCONFIGDIR="${FIG2_RUN_DIR}/runtime_cache/matplotlib"
fi

fig2_check_env() {
    "${FIG2_ENV}/bin/python" "${FIG2_ROOT}/jobs/figure2/check_env.py" \
        --root "${FIG2_ROOT}" --env "${FIG2_ENV}" "$@"
}

fig2_require_gpu() {
    "${FIG2_ENV}/bin/python" - <<'PY'
from pathlib import Path
import os
import sys
import torch
import transformers
from packaging.version import Version
if Path(sys.prefix).resolve() != Path(os.environ['FIG2_ENV']).resolve():
    raise SystemExit(f'Wrong environment: {sys.prefix}')
if Version(torch.__version__.split('+')[0]) < Version('2.4'):
    raise SystemExit('PyTorch >=2.4 is required')
if transformers.__version__ != '4.57.3':
    raise SystemExit('Figure 2 currently requires transformers==4.57.3; inspect lab before changing packages.')
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; CPU fallback is disabled')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('This experiment requires bfloat16 support')
print('environment=', sys.prefix)
print('torch=', torch.__version__, 'transformers=', transformers.__version__)
print('gpu=', torch.cuda.get_device_name(0), 'CUDA=', torch.version.cuda)
PY
}
