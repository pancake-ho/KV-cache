#!/usr/bin/bash
#SBATCH --job-name=sparge-install
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/sparge-install-%j.out
#SBATCH --error=logs/sparge-install-%j.err

set -Eeuo pipefail

readonly SPARGE_REPO="https://github.com/thu-ml/SpargeAttn.git"
readonly SPARGE_COMMIT="ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a"
readonly TARGET="/data/${USER}/third_party/SpargeAttn"

readonly SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

REPO_ROOT="$(
    git -C "${SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null
)" || {
    echo "[ERROR] Could not resolve repository root." >&2
    exit 2
}
readonly REPO_ROOT

cd "${REPO_ROOT}"

if [[ "$(git branch --show-current)" != "exp/sparkv-test" ]]; then
    echo "[ERROR] install script must be submitted from exp/sparkv-test." >&2
    exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "lab" ]]; then
    CONDA_SH=""

    for candidate in \
        "/data/${USER}/anaconda3/etc/profile.d/conda.sh" \
        "/data/${USER}/miniconda3/etc/profile.d/conda.sh"; do
        if [[ -f "${candidate}" ]]; then
            CONDA_SH="${candidate}"
            break
        fi
    done

    if [[ -n "${CONDA_SH}" ]]; then
        # shellcheck disable=SC1090
        source "${CONDA_SH}"
        conda activate lab
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate lab
    else
        echo "[ERROR] Could not activate conda env lab." >&2
        exit 3
    fi
fi

mkdir -p \
    "/data/${USER}/third_party" \
    "logs"

if [[ ! -d "${TARGET}/.git" ]]; then
    git clone \
        "${SPARGE_REPO}" \
        "${TARGET}"
fi

git -C "${TARGET}" fetch origin
git -C "${TARGET}" checkout \
    --detach \
    "${SPARGE_COMMIT}"

python -m pip install \
    -r baseline/sparkv/requirements.txt

python -m pip install \
    -e "${TARGET}"

python - <<'PY'
import torch
import bitarray
from spas_sage_attn import (
    block_sparse_sage2_attn_cuda,
)

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("bitarray:", getattr(bitarray, "__version__", "unknown"))
print("available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")

print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("SpargeAttention import: OK")
PY

python -m pip freeze \
    | grep -E \
        '^(torch|transformers|triton|bitarray|nvidia-ml-py|spas|sage|sparge)' \
    || true
