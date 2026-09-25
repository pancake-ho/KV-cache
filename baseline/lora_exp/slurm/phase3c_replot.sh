#!/usr/bin/env bash
#SBATCH -J lora-p3c-replot
#SBATCH -p batch_eebme_ugrad
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH -o runs/phase3c_replot/slurm-%j.out
#SBATCH -e runs/phase3c_replot/slurm-%j.err

set -euo pipefail

PROJECT_DIR="/data/${USER}/repos/lab/kv_cache/baseline/lora_exp"

cd "${PROJECT_DIR}"
mkdir -p runs/phase3c_replot

source /data/"$USER"/anaconda3/etc/profile.d/conda.sh
conda activate lab

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/${USER}_mpl_phase3c_replot_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

echo "============================================================"
echo "[SBATCH-START]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"
echo "============================================================"

python scripts/16_plot_phase3c_professor.py \
  --config configs/phase3c_cross_domain.yaml

echo "============================================================"
echo "[SBATCH-END]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "============================================================"
