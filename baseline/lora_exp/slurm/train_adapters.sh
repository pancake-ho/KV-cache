#!/usr/bin/env bash

#SBATCH -J lora-p2-train
#SBATCH -p batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH --time=1-0
#SBATCH -o runs/train/slurm-%j.out
#SBATCH -e runs/train/slurm-%j.err

set -euo pipefail


PROJECT_DIR="/data/surt321/repos/lab/kv_cache/baseline/lora_exp"

WORK_DIR="/local_datasets/${USER}/lora_exp/phase2_${SLURM_JOB_ID}"

HF_ROOT="/local_datasets/${USER}/hf_cache/lora_phase2_${SLURM_JOB_ID}"


echo "============================================================"
echo "[SBATCH-START]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID:-NA}"
echo "host=$(hostname)"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "WORK_DIR=${WORK_DIR}"
echo "============================================================"


cd "${PROJECT_DIR}"

mkdir -p runs/train
mkdir -p "$(dirname "${WORK_DIR}")"
mkdir -p "${HF_ROOT}"


if [[ -e "${WORK_DIR}" ]]; then
    echo "[FATAL] WORK_DIR already exists:"
    echo "${WORK_DIR}"
    exit 1
fi


source /data/"$USER"/anaconda3/etc/profile.d/conda.sh
conda activate lab


export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HOME="${HF_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_ROOT}/hub"
export HF_DATASETS_CACHE="${HF_ROOT}/datasets"

export TOKENIZERS_PARALLELISM=false


echo
echo "[ENV]"

which python
python --version

echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"

echo
echo "[GIT STATUS]"
git status --short || true


echo
echo "[GPU]"

nvidia-smi

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable; CPU fallback forbidden."
    )

print("GPU       :", torch.cuda.get_device_name(0))
print("Torch     :", torch.__version__)
print("CUDA      :", torch.version.cuda)
print("BF16      :", torch.cuda.is_bf16_supported())
PY


echo
echo "[PACKAGE CHECK]"

python - <<'PY'
import bitsandbytes
import datasets
import peft
import transformers

print("bitsandbytes :", bitsandbytes.__version__)
print("datasets     :", datasets.__version__)
print("peft         :", peft.__version__)
print("transformers :", transformers.__version__)
PY


echo
echo "[LOCAL STORAGE]"

df -h /local_datasets || true

FREE_KB=$(df -Pk /local_datasets | awk 'NR==2 {print $4}')
MIN_FREE_KB=$((20 * 1024 * 1024))

if [[ "${FREE_KB}" -lt "${MIN_FREE_KB}" ]]; then
    echo "[FATAL] Less than 20 GiB free local space."
    exit 2
fi


echo
echo "[RUN PHASE 2 TRAINING]"

python scripts/02_train_adapters.py \
    --model-config configs/model.yaml \
    --lora-config configs/lora.yaml \
    --train-config configs/train.yaml \
    --work-dir "${WORK_DIR}"


echo
echo "============================================================"
echo "[SBATCH-END]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "============================================================"