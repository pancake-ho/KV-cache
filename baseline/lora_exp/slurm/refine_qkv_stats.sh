#!/usr/bin/env bash

#SBATCH -J lora-p3-stat
#SBATCH -p batch_eebme_ugrad
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o runs/qkv_stat/slurm-%j.out
#SBATCH -e runs/qkv_stat/slurm-%j.err

set -euo pipefail


PROJECT_DIR="/data/surt321/repos/lab/kv_cache/baseline/lora_exp"


cd "${PROJECT_DIR}"

mkdir -p runs/qkv_stat


source /data/"$USER"/anaconda3/etc/profile.d/conda.sh
conda activate lab


export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export MPLCONFIGDIR="/tmp/${USER}_mpl_qkvstat_${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}"


echo "============================================================"
echo "[SBATCH-START]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"
echo "============================================================"


echo
echo "[PACKAGE CHECK]"

python - <<'PY'
import matplotlib
import numpy
import pandas
import yaml

print("matplotlib :", matplotlib.__version__)
print("numpy      :", numpy.__version__)
print("pandas     :", pandas.__version__)
print("pyyaml     :", yaml.__version__)
PY


echo
echo "[RUN PHASE 3A REFINED STATISTICS]"

python scripts/07_refine_qkv_stats.py \
    --config configs/characterize.yaml


echo
echo "[OUTPUT TREE]"

find \
  /data/${USER}/models/lora_exp/phase3_qkv_medical_control_seed2026/refined_stats \
  -maxdepth 1 \
  -type f \
  -printf '%f\n' \
  | sort


echo
echo "[FIGURE]"

ls -lh \
  /data/${USER}/models/lora_exp/phase3_qkv_medical_control_seed2026/figures/phase3a_overview.png


echo
echo "============================================================"
echo "[SBATCH-END]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "============================================================"