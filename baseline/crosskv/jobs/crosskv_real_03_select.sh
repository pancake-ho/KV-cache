#!/bin/bash
#SBATCH --job-name=xkv-select
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=08:00:00
#SBATCH --output=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.out
#SBATCH --error=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.err

set -euo pipefail

ROOT="/data/surt321/repos/lab/kv_cache/baseline/crosskv"
WORK="${ROOT}/work/crosskv_qwen3_17b_to_4b"

cd "$ROOT"

source /data/surt321/anaconda3/etc/profile.d/conda.sh
conda activate lab

CUDA_VISIBLE_DEVICES=0 \
python -m xmodel_kv.cli.select \
  --source "${WORK}/source" \
  --target "${WORK}/target" \
  --output "${WORK}/selection_k12.json" \
  --top-k 12 \
  --ridge 0 \
  --device cuda:0 \
  --max-observations 32768

python - "${WORK}/selection_k12.json" <<'PY'
import json
import sys

x = json.load(open(sys.argv[1]))

print("selection N =", x["num_observations"])
print("top-k       =", x["top_k"])
print("target layers =", len(x["selected_layers"]))

for i, layers in enumerate(x["selected_layers"]):
    print(
        f"target {i:02d}:",
        layers,
    )
PY
