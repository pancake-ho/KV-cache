#!/bin/bash
#SBATCH --job-name=xkv-real-batch
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=08:00:00
#SBATCH --output=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.out
#SBATCH --error=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.err

set -euo pipefail

ROOT="/data/surt321/repos/lab/kv_cache/baseline/crosskv"

cd "${ROOT}"

source \
/data/surt321/anaconda3/etc/profile.d/conda.sh

conda activate lab

export HF_HOME=/data/surt321/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if ! nvidia-smi >/dev/null 2>&1
then
    echo "ERROR: GPU unavailable"
    exit 1
fi

OUT="outputs/real_handoff_batch"

mkdir -p "${OUT}"

echo "=============================================="
echo "CrossKV REAL HotpotQA batch"
echo "=============================================="

echo "DATE=$(date --iso-8601=seconds)"
echo "HOST=$(hostname)"
echo "JOB=${SLURM_JOB_ID}"

echo
echo "[Git]"

git branch --show-current
git rev-parse HEAD
git status --short

echo
nvidia-smi

CUDA_VISIBLE_DEVICES=0 \
python scripts/run_hotpot_crosskv_batch.py \
    --dataset \
        datasets/longbench/hotpotqa_e.jsonl \
    --source-model \
        Qwen/Qwen3-1.7B \
    --target-model \
        Qwen/Qwen3-4B \
    --mapper \
        work/crosskv_qwen3_17b_to_4b_longcal4k/mapper_k12 \
    --target-lengths \
        2200 4096 8192 \
    --per-length \
        4 \
    --max-new-tokens \
        64 \
    --selection-output \
        "${OUT}/selected_${SLURM_JOB_ID}.json" \
    --output \
        "${OUT}/results_${SLURM_JOB_ID}.jsonl" \
    --summary \
        "${OUT}/summary_${SLURM_JOB_ID}.json"

echo
echo "=============================================="
echo "DONE"
echo "=============================================="

cat "${OUT}/summary_${SLURM_JOB_ID}.json"

