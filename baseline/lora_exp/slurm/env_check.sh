#!/usr/bin/env bash

#SBATCH -J lora-p0-check
#SBATCH -p batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH --time=1-0
#SBATCH -o runs/env_check/slurm-%j.out
#SBATCH -e runs/env_check/slurm-%j.err

set -euo pipefail


PROJECT_DIR="/data/surt321/repos/lab/kv_cache/baseline/lora-exp"

echo "============================================================"
echo "[SBATCH-START]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID:-NA}"
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "============================================================"


cd "${PROJECT_DIR}"

# ------------------------------------------------------------
# Conda
# ------------------------------------------------------------

source /data/"$USER"/anaconda3/etc/profile.d/conda.sh
conda activate lab


echo
echo "[ENV]"
which python
python --version

echo
echo "[GPU]"
nvidia-smi


echo
echo "[PACKAGE VERSIONS]"

python - <<'PY'
import torch
import transformers

print("torch        :", torch.__version__)
print("cuda         :", torch.version.cuda)
print("transformers :", transformers.__version__)

try:
    import peft
    print("peft         :", peft.__version__)
except Exception as exc:
    raise RuntimeError(
        "PEFT is unavailable. Install/repair PEFT before continuing."
    ) from exc

try:
    import yaml
    print("pyyaml       :", yaml.__version__)
except Exception as exc:
    raise RuntimeError(
        "PyYAML is unavailable. Install pyyaml before continuing."
    ) from exc
PY


echo
echo "[RUN PHASE 0]"

python scripts/00_check_env.py \
    --model-config configs/model.yaml \
    --lora-config configs/lora.yaml \
    --seed 2026 \
    --output-dir runs/env_check


echo
echo "============================================================"
echo "[SBATCH-END]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "============================================================"