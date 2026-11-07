#!/bin/bash
#SBATCH --job-name=xkv-real-handoff
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=02:00:00
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


echo "============================================================"
echo "REAL CROSSKV HANDOFF"
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


python - <<'PY'
import torch

print(
    "CUDA available:",
    torch.cuda.is_available(),
)

if not torch.cuda.is_available():

    raise SystemExit(
        "ERROR: CUDA unavailable. "
        "Do not continue on CPU."
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0),
)

print(
    "VRAM GiB:",
    torch.cuda.get_device_properties(
        0
    ).total_memory / 1024**3,
)
PY


OUTPUT_DIR="outputs/real_handoff"

mkdir -p "${OUTPUT_DIR}"


CUDA_VISIBLE_DEVICES=0 \
python scripts/run_hotpot_crosskv_handoff.py \
    --source-model Qwen/Qwen3-1.7B \
    --target-model Qwen/Qwen3-4B \
    --mapper \
        work/crosskv_qwen3_17b_to_4b/mapper_k12 \
    --prompt-file \
        inputs/hotpotqa_index83_prompt.txt \
    --meta-file \
        inputs/hotpotqa_index83_meta.json \
    --output \
        "${OUTPUT_DIR}/hotpot_index83_${SLURM_JOB_ID}.json" \
    --max-new-tokens 64


echo
echo "============================================================"
echo "DONE"
echo "============================================================"

echo "DATE=$(date --iso-8601=seconds)"
