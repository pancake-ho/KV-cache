#!/usr/bin/env bash

#SBATCH -J lora-p3-probe
#SBATCH -p batch_eebme_ugrad
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o runs/probe/slurm-%j.out
#SBATCH -e runs/probe/slurm-%j.err

set -euo pipefail


PROJECT_DIR="/data/surt321/repos/lab/kv_cache/baseline/lora_exp"

WORK_DIR="/local_datasets/${USER}/lora_exp/phase3_probe_${SLURM_JOB_ID}"

HF_ROOT="/local_datasets/${USER}/hf_cache/lora_phase3_probe_${SLURM_JOB_ID}"


cd "${PROJECT_DIR}"

mkdir -p runs/probe
mkdir -p "$(dirname "${WORK_DIR}")"
mkdir -p "${HF_ROOT}"


if [[ -e "${WORK_DIR}" ]]; then
    echo "[FATAL] WORK_DIR exists:"
    echo "${WORK_DIR}"
    exit 1
fi


source /data/"$USER"/anaconda3/etc/profile.d/conda.sh
conda activate lab


export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HOME="${HF_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_ROOT}/hub"
export HF_DATASETS_CACHE="${HF_ROOT}/datasets"

export TOKENIZERS_PARALLELISM=false


echo "============================================================"
echo "[SBATCH-START]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit=$(git rev-parse HEAD)"
echo "============================================================"


python scripts/04_build_probes.py \
    --config configs/characterize.yaml \
    --model-config configs/model.yaml \
    --work-dir "${WORK_DIR}"


echo
echo "[PROBE MANIFEST]"

cat data/probes/phase3_medical_control_seed2026_manifest.json


echo
echo "============================================================"
echo "[SBATCH-END]"
echo "timestamp=$(date --iso-8601=seconds)"
echo "============================================================"