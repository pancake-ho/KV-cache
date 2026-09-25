#!/usr/bin/env bash
#SBATCH -J lora-p3c-data
#SBATCH -p batch_eebme_ugrad
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -o runs/phase3c_data/slurm-%j.out
#SBATCH -e runs/phase3c_data/slurm-%j.err

set -euo pipefail

PROJECT_DIR="/data/${USER}/repos/lab/kv_cache/baseline/lora_exp"
CACHE_DIR="/local_datasets/${USER}/hf_cache/phase3c_${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
mkdir -p runs/phase3c_data "${CACHE_DIR}"

source /data/"$USER"/anaconda3/etc/profile.d/conda.sh
conda activate lab
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/hub"
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "[SBATCH-START] $(date --iso-8601=seconds)"
echo "job=${SLURM_JOB_ID} host=$(hostname)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"
echo "============================================================"

python scripts/11_prepare_cross_domain.py \
  --config configs/phase3c_cross_domain.yaml \
  --cache-dir "${CACHE_DIR}"

cat /data/${USER}/datasets/lora_exp/cross_domain_v1/manifest.json

echo "[SBATCH-END] $(date --iso-8601=seconds)"
