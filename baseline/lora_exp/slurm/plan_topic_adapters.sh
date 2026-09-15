#!/usr/bin/env bash
#SBATCH -J lora-p3b-plan
#SBATCH -p batch_eebme_ugrad
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o runs/topic_plan/slurm-%j.out
#SBATCH -e runs/topic_plan/slurm-%j.err

set -euo pipefail

PROJECT_DIR="/data/surt321/repos/lab/kv_cache/baseline/lora_exp"
WORK_DIR="/local_datasets/${USER}/lora_exp/phase3b_topic_plan_${SLURM_JOB_ID}"
HF_ROOT="/local_datasets/${USER}/hf_cache/lora_phase3b_topic_plan_${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"

mkdir -p runs/topic_plan
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

echo "============================================================"
echo "[SBATCH-START]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"
echo "============================================================"

python scripts/08_train_topic_adapters.py \
    --config configs/topic_userpair.yaml \
    --work-dir "${WORK_DIR}" \
    --plan-only

echo
echo "[TOPIC PLAN]"
cat /data/${USER}/models/lora_exp/phase3b_topic_adapters_seed2026/topic_plan.json

echo "============================================================"
echo "[SBATCH-END]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "============================================================"
