#!/usr/bin/env bash
#SBATCH -J lora-p3c-summary
#SBATCH -p batch_eebme_ugrad
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o runs/phase3c_summary/slurm-%j.out
#SBATCH -e runs/phase3c_summary/slurm-%j.err

set -euo pipefail
PROJECT_DIR="/data/${USER}/repos/lab/kv_cache/baseline/lora_exp"
cd "${PROJECT_DIR}"
mkdir -p runs/phase3c_summary

source /data/"$USER"/anaconda3/etc/profile.d/conda.sh
conda activate lab
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/${USER}_mpl_phase3c_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

python scripts/15_summarize_cross_domain.py \
  --config configs/phase3c_cross_domain.yaml
