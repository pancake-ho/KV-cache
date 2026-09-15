#!/usr/bin/env bash

#SBATCH -J lora-p1-data
#SBATCH -p batch_eebme_ugrad
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-0
#SBATCH -o runs/data/slurm-%j.out
#SBATCH -e runs/data/slurm-%j.err

set -euo pipefail


PROJECT_DIR="/data/surt321/repos/lab/kv_cache/baseline/lora-exp"

WORK_DIR="/local_datasets/${USER}/lora-exp/phase1_${SLURM_JOB_ID}"

HF_ROOT="/local_datasets/${USER}/hf_cache/lora_phase1_${SLURM_JOB_ID}"

ARCHIVE_DIR="/data/${USER}/datasets/lora-exp/archives"

MANIFEST_DIR="${PROJECT_DIR}/data/manifests"


echo "============================================================"
echo "[SBATCH-START]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID:-NA}"
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "============================================================"


cd "${PROJECT_DIR}"

mkdir -p runs/data
mkdir -p "${WORK_DIR}"
mkdir -p "${HF_ROOT}"
mkdir -p "${ARCHIVE_DIR}"
mkdir -p "${MANIFEST_DIR}"


# ------------------------------------------------------------
# Conda
# ------------------------------------------------------------

source /data/"$USER"/anaconda3/etc/profile.d/conda.sh
conda activate lab


echo
echo "[ENV]"
which python
python --version
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"

echo
echo "[GIT STATUS]"
git status --short || true


# ------------------------------------------------------------
# Hugging Face cache must stay on compute-node local storage.
# ------------------------------------------------------------

export HF_HOME="${HF_ROOT}"
export HF_DATASETS_CACHE="${HF_ROOT}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_ROOT}/hub"

export TOKENIZERS_PARALLELISM=false


echo
echo "[LOCAL STORAGE]"
echo "WORK_DIR=${WORK_DIR}"
echo "HF_HOME=${HF_HOME}"
df -h /local_datasets || true


echo
echo "[PACKAGE CHECK]"

python - <<'PY'
import datasets
import huggingface_hub
import pyarrow
import transformers
import yaml

print("datasets        :", datasets.__version__)
print("huggingface_hub :", huggingface_hub.__version__)
print("pyarrow         :", pyarrow.__version__)
print("transformers    :", transformers.__version__)
print("pyyaml          :", yaml.__version__)
PY


echo
echo "[RUN PHASE 1]"

python scripts/01_prepare_data.py \
    --config configs/data.yaml \
    --work-dir "${WORK_DIR}" \
    --archive-dir "${ARCHIVE_DIR}" \
    --project-manifest-dir data/manifests


echo
echo "[ARCHIVES]"

ls -lh "${ARCHIVE_DIR}" | tail -20


echo
echo "[MANIFESTS]"

ls -lh "${MANIFEST_DIR}"


echo
echo "[LOCAL WORK SIZE]"

du -sh "${WORK_DIR}" || true


echo
echo "============================================================"
echo "[SBATCH-END]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "============================================================"