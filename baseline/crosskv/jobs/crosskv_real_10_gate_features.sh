#!/bin/bash
#SBATCH --job-name=xkv-gate-feat
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=04:00:00
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

CUDA_VISIBLE_DEVICES=0 \
python scripts/collect_source_gate_features.py \
    --dataset \
        datasets/longbench/hotpotqa_e.jsonl \
    --results \
        "${OUT}/results_rescored.jsonl" \
    --source-model \
        Qwen/Qwen3-1.7B \
    --output \
        "${OUT}/source_gate_features.jsonl"

