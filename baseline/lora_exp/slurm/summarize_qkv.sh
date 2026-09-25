#!/usr/bin/env bash

#SBATCH -J lora-p3-summary
#SBATCH -p batch_eebme_ugrad
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o runs/qkv_summary/slurm-%j.out
#SBATCH -e runs/qkv_summary/slurm-%j.err

set -euo pipefail


PROJECT_DIR="/data/surt321/repos/lab/kv_cache/baseline/lora_exp"


cd "${PROJECT_DIR}"

mkdir -p runs/qkv_summary


source /data/"$USER"/anaconda3/etc/profile.d/conda.sh
conda activate lab


export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export MPLCONFIGDIR="/tmp/${USER}_mpl_${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}"


echo "============================================================"
echo "[SBATCH-START]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "============================================================"


python scripts/06_summarize_qkv.py \
    --config configs/characterize.yaml


echo
echo "============================================================"
echo "[SBATCH-END]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "============================================================"