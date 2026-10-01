#!/usr/bin/env bash
# Run once. Clones the existing CUDA environment and leaves it unchanged.

set -euo pipefail
FIG2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
FIG2_CONDA_ROOT="${FIG2_CONDA_ROOT:-/data/${USER}/anaconda3}"
FIG2_BASE_ENV="${FIG2_BASE_ENV:-${FIG2_CONDA_ROOT}/envs/lab}"
FIG2_ENV="${FIG2_ENV:-${FIG2_CONDA_ROOT}/envs/crosskv_fig2}"

source "${FIG2_CONDA_ROOT}/etc/profile.d/conda.sh"
if [[ "${FIG2_BASE_ENV}" == "${FIG2_ENV}" ]]; then
    echo 'Use a separate destination environment.' >&2
    exit 2
fi
if [[ ! -d "${FIG2_ENV}" ]]; then
    conda create --yes --prefix "${FIG2_ENV}" --clone "${FIG2_BASE_ENV}"
fi
conda activate "${FIG2_ENV}"
python - <<'PY'
import torch
from packaging.version import Version
if Version(torch.__version__.split('+')[0]) < Version('2.4'):
    raise SystemExit('The cloned environment needs CUDA-enabled PyTorch >=2.4. Choose a suitable base environment.')
if torch.version.cuda is None:
    raise SystemExit('The cloned environment contains CPU-only PyTorch. Choose your working CUDA environment.')
print('Preserving PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda)
PY
python -m pip install -r "${FIG2_ROOT}/requirements-figure2.txt"
python -m pip install --no-deps -e "${FIG2_ROOT}"
printf 'Ready: FIG2_ENV=%s\n' "${FIG2_ENV}"
