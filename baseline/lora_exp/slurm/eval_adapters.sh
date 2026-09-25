#!/usr/bin/env bash

#SBATCH -J lora-p2-eval
#SBATCH -p batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH --time=06:00:00
#SBATCH -o runs/eval/slurm-%j.out
#SBATCH -e runs/eval/slurm-%j.err

set -euo pipefail


PROJECT_DIR="/data/surt321/repos/lab/kv_cache/baseline/lora_exp"

WORK_DIR="/local_datasets/${USER}/lora_exp/phase2_eval_${SLURM_JOB_ID}"

HF_ROOT="/local_datasets/${USER}/hf_cache/lora_phase2_eval_${SLURM_JOB_ID}"


cd "${PROJECT_DIR}"

mkdir -p runs/eval
mkdir -p "$(dirname "${WORK_DIR}")"
mkdir -p "${HF_ROOT}"


if [[ -e "${WORK_DIR}" ]]; then
    echo "[FATAL] WORK_DIR exists:"
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


echo "============================================================"
echo "[SBATCH-START]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID:-NA}"
echo "host=$(hostname)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"
echo "============================================================"


nvidia-smi


python scripts/03_eval_adapters.py \
    --model-config configs/model.yaml \
    --train-config configs/train.yaml \
    --work-dir "${WORK_DIR}"


echo
echo "============================================================"
echo "[SBATCH-END]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "============================================================"