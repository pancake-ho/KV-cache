#!/bin/bash
#SBATCH --job-name=xkv-full-curve
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

OUT="outputs/real_handoff_full_le10k"

mkdir -p "${OUT}"

echo "============================================================"
echo "CrossKV FULL REAL OPERATING CURVE"
echo "============================================================"

echo "DATE=$(date --iso-8601=seconds)"
echo "HOST=$(hostname)"
echo "JOB=${SLURM_JOB_ID}"

echo
echo "[Git]"

git branch --show-current
git rev-parse HEAD
git status --short

echo
echo "[GPU]"

nvidia-smi

CUDA_VISIBLE_DEVICES=0 \
python scripts/run_hotpot_crosskv_full_curve.py \
    --dataset \
        datasets/longbench/hotpotqa_e.jsonl \
    --source-model \
        Qwen/Qwen3-1.7B \
    --target-model \
        Qwen/Qwen3-4B \
    --mapper \
        work/crosskv_qwen3_17b_to_4b_longcal4k/mapper_k12 \
    --max-context-tokens \
        10000 \
    --max-new-tokens \
        64 \
    --eligible-output \
        "${OUT}/eligible.json" \
    --output \
        "${OUT}/results.jsonl" \
    --summary \
        "${OUT}/summary.json"

echo
echo "============================================================"
echo "DONE"
echo "============================================================"

cat "${OUT}/summary.json"

