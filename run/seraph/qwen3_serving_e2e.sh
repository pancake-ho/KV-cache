#!/usr/bin/env bash
#SBATCH --job-name=kv-top-e2e
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=moana-u1,moana-u2,moana-u3,moana-u4,moana-u8,moana-y4
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
: "${SLURM_JOB_ID:?submit through Slurm}"
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
test -f proposed/validation/serving_e2e.py

conda_base=$(conda info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate lab
set -u

python_bin=$(command -v python)
probe='import sys; import torch; sys.exit(0 if torch.cuda.is_available() else 1)'
if ! "$python_bin" -c "$probe" && [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    if env -u LD_LIBRARY_PATH "$python_bin" -c "$probe"; then
        echo "Using the validated CUDA environment without the inherited LD_LIBRARY_PATH"
        unset LD_LIBRARY_PATH
    fi
fi
exec "$python_bin" -u -m proposed.validation.serving_e2e
