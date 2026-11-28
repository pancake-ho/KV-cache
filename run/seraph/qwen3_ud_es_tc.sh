#!/usr/bin/env bash
#SBATCH --job-name=kv-comm
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=1-0
#SBATCH --output=results/comm/%x-%j.out
#SBATCH --error=results/comm/%x-%j.err

set -eo pipefail
: "${SLURM_JOB_ID:?submit through Slurm}"
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"

conda_base=$(conda info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate lab
set -u

exec unshare --user --map-root-user --mount --net --propagation private \
    python -u -c 'from proposed.communication.private_network import run; run()'
