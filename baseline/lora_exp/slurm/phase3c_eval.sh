#!/usr/bin/env bash
#SBATCH -J lora-p3c-eval
#SBATCH -p batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH --time=08:00:00
#SBATCH -o runs/phase3c_eval/slurm-%j.out
#SBATCH -e runs/phase3c_eval/slurm-%j.err

set -euo pipefail
PROJECT_DIR="/data/${USER}/repos/lab/kv_cache/baseline/lora_exp"
cd "${PROJECT_DIR}"
mkdir -p runs/phase3c_eval

source /data/"$USER"/anaconda3/etc/profile.d/conda.sh
conda activate lab
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

echo "[SBATCH-START] $(date --iso-8601=seconds) host=$(hostname)"
nvidia-smi
python scripts/13_eval_cross_domain.py \
  --config configs/phase3c_cross_domain.yaml
echo "[SBATCH-END] $(date --iso-8601=seconds)"
