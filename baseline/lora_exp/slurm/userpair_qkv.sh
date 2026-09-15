#!/usr/bin/env bash
#SBATCH -J lora-p3b-pairs
#SBATCH -p batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH --time=12:00:00
#SBATCH -o runs/userpair/slurm-%j.out
#SBATCH -e runs/userpair/slurm-%j.err

set -euo pipefail

PROJECT_DIR="/data/surt321/repos/lab/kv_cache/baseline/lora_exp"
WORK_DIR="/local_datasets/${USER}/lora_exp/phase3b_userpair_${SLURM_JOB_ID}"
HF_ROOT="/local_datasets/${USER}/hf_cache/lora_phase3b_userpair_${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"

mkdir -p runs/userpair
mkdir -p "$(dirname "${WORK_DIR}")"
mkdir -p "${HF_ROOT}"

if [[ -e "${WORK_DIR}" ]]; then
    echo "[FATAL] WORK_DIR exists: ${WORK_DIR}"
    exit 1
fi

source /data/"$USER"/anaconda3/etc/profile.d/conda.sh
conda activate lab

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_ROOT}/hub"
export HF_DATASETS_CACHE="${HF_ROOT}/datasets"
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="/tmp/${USER}_mpl_phase3b_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

echo "============================================================"
echo "[SBATCH-START]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"
echo "============================================================"

nvidia-smi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; CPU fallback forbidden.")
print("GPU   :", torch.cuda.get_device_name(0))
print("Torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)
PY

python scripts/10_userpair_qkv.py \
    --config configs/topic_userpair.yaml \
    --work-dir "${WORK_DIR}"

echo "============================================================"
echo "[SBATCH-END]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "============================================================"
