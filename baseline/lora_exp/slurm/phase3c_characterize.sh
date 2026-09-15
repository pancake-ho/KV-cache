#!/usr/bin/env bash
#SBATCH -J lora-p3c-qkv
#SBATCH -p batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH --time=12:00:00
#SBATCH -o runs/phase3c_qkv/slurm-%j.out
#SBATCH -e runs/phase3c_qkv/slurm-%j.err

set -euo pipefail
PROJECT_DIR="/data/${USER}/repos/lab/kv_cache/baseline/lora_exp"
cd "${PROJECT_DIR}"
mkdir -p runs/phase3c_qkv

source /data/"$USER"/anaconda3/etc/profile.d/conda.sh
conda activate lab
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "[SBATCH-START] $(date --iso-8601=seconds) host=$(hostname)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"
echo "============================================================"
nvidia-smi
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; CPU fallback forbidden.")
print("GPU:", torch.cuda.get_device_name(0))
PY

python scripts/14_characterize_cross_domain.py \
  --config configs/phase3c_cross_domain.yaml

echo "[SBATCH-END] $(date --iso-8601=seconds)"
