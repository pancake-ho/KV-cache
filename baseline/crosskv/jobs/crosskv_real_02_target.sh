#!/bin/bash
#SBATCH --job-name=xkv4-tgt
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.out
#SBATCH --error=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.err

set -euo pipefail

ROOT="/data/surt321/repos/lab/kv_cache/baseline/crosskv"

cd "$ROOT"

source /data/surt321/anaconda3/etc/profile.d/conda.sh
conda activate lab

export HF_HOME=/data/surt321/hf_cache
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")

print(torch.cuda.get_device_name(0))
PY

CUDA_VISIBLE_DEVICES=0 \
python -m xmodel_kv.cli.extract \
  --model Qwen/Qwen3-4B \
  --tokens work/crosskv_qwen3_17b_to_4b/tokens.npy \
  --output work/crosskv_qwen3_17b_to_4b/target \
  --role target \
  --stride 4 \
  --device-map cuda:0 \
  --attn-implementation sdpa

cat work/crosskv_qwen3_17b_to_4b/target/metadata.json
du -sh work/crosskv_qwen3_17b_to_4b/target
