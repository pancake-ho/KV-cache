#!/bin/bash
#SBATCH --job-name=xkv-fit
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.out
#SBATCH --error=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.err

set -euo pipefail

ROOT="/data/surt321/repos/lab/kv_cache/baseline/crosskv"
WORK="${ROOT}/work/crosskv_qwen3_17b_to_4b"

cd "$ROOT"

source /data/surt321/anaconda3/etc/profile.d/conda.sh
conda activate lab

CUDA_VISIBLE_DEVICES=0 \
python -m xmodel_kv.cli.fit \
  --source "${WORK}/source" \
  --target "${WORK}/target" \
  --selection "${WORK}/selection_k12.json" \
  --output "${WORK}/mapper_k12" \
  --ridge 0.01 \
  --device cuda:0 \
  --max-observations 128000 \
  --weight-dtype float32

echo
echo "[MAPPER CONFIG]"

cat "${WORK}/mapper_k12/config.json"

echo
echo "[MAPPER SIZE]"

du -sh "${WORK}/mapper_k12"
