#!/usr/bin/env bash

#SBATCH -J lora-p3-qkv
#SBATCH -p batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH --time=12:00:00
#SBATCH -o runs/qkv/slurm-%j.out
#SBATCH -e runs/qkv/slurm-%j.err

set -euo pipefail


PROJECT_DIR="/data/surt321/repos/lab/kv_cache/baseline/lora_exp"

HF_ROOT="/local_datasets/${USER}/hf_cache/lora_phase3_qkv_${SLURM_JOB_ID}"


cd "${PROJECT_DIR}"

mkdir -p runs/qkv
mkdir -p "${HF_ROOT}"


source /data/"$USER"/anaconda3/etc/profile.d/conda.sh
conda activate lab


export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HOME="${HF_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_ROOT}/hub"

export TOKENIZERS_PARALLELISM=false


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
    raise RuntimeError(
        "CUDA unavailable; CPU fallback forbidden."
    )

print("GPU  :", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__)
print("CUDA :", torch.version.cuda)
print("BF16 :", torch.cuda.is_bf16_supported())
PY


echo
echo "[RUN QKV CHARACTERIZATION]"

python scripts/05_characterize_qkv.py \
    --config configs/characterize.yaml \
    --model-config configs/model.yaml


echo
echo "============================================================"
echo "[SBATCH-END]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "============================================================"