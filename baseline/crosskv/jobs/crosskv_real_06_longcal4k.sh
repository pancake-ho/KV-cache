#!/bin/bash
#SBATCH --job-name=xkv-longcal4k
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.out
#SBATCH --error=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.err

set -euo pipefail

ROOT="/data/surt321/repos/lab/kv_cache/baseline/crosskv"

cd "${ROOT}"

source /data/surt321/anaconda3/etc/profile.d/conda.sh
conda activate lab

export HF_HOME=/data/surt321/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SOURCE_MODEL="Qwen/Qwen3-1.7B"
TARGET_MODEL="Qwen/Qwen3-4B"

DATASET="/data/surt321/datasets/fineweb-edu/sample/10BT/000_00000.parquet"

WORK="${ROOT}/work/crosskv_qwen3_17b_to_4b_longcal4k"

mkdir -p "${WORK}"
mkdir -p "${ROOT}/outputs/real_handoff_longcal4k"
mkdir -p "${ROOT}/jobs/logs"


echo "============================================================"
echo "CrossKV real Stage 06"
echo "4K-context calibrated mapper"
echo "============================================================"

echo "START=$(date --iso-8601=seconds)"
echo "JOB=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"

echo
echo "[Git]"

git branch --show-current
git rev-parse HEAD
git status --short


echo
echo "[GPU]"

python - <<'PY'
import torch

print(
    "CUDA available:",
    torch.cuda.is_available(),
)

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: CUDA unavailable. "
        "CPU fallback is not allowed."
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

nvidia-smi


# ============================================================
# 1. Prepare 125 × 4096
#
# 125 × 4096 / stride 4
# = 128,000 observations
#
# Same total N as old 500 × 1024 mapper.
# ============================================================

echo
echo "============================================================"
echo "[1] PREPARE 4K CALIBRATION TOKENS"
echo "============================================================"

if [[ ! -f "${WORK}/.prepare.done" ]]
then

    rm -f \
        "${WORK}/tokens.npy" \
        "${WORK}/tokens.json"

    python -m xmodel_kv.cli.prepare \
        --dataset "${DATASET}" \
        --tokenizer "${SOURCE_MODEL}" \
        --output "${WORK}/tokens.npy" \
        --text-field text \
        --num-sequences 125 \
        --sequence-length 4096

    python - "${WORK}/tokens.npy" <<'PY'
import sys
import numpy as np

x = np.load(
    sys.argv[1],
    mmap_mode="r",
)

print("tokens shape =", x.shape)

assert x.shape == (125, 4096)
PY

    touch "${WORK}/.prepare.done"

else

    echo "Prepare already completed."

fi


# ============================================================
# 2. Source extraction
# ============================================================

echo
echo "============================================================"
echo "[2] SOURCE EXTRACTION"
echo "============================================================"

if [[ ! -f "${WORK}/.source.done" ]]
then

    # extraction.py writes metadata before completing all
    # sequences, so metadata.json alone is NOT a completion
    # marker. Always recreate if our explicit .done is absent.
    rm -rf "${WORK}/source"

    CUDA_VISIBLE_DEVICES=0 \
    python -m xmodel_kv.cli.extract \
        --model "${SOURCE_MODEL}" \
        --tokens "${WORK}/tokens.npy" \
        --output "${WORK}/source" \
        --role source \
        --stride 4 \
        --device-map cuda:0 \
        --attn-implementation sdpa

    touch "${WORK}/.source.done"

else

    echo "Source extraction already completed."

fi


# ============================================================
# 3. Target extraction
# ============================================================

echo
echo "============================================================"
echo "[3] TARGET EXTRACTION"
echo "============================================================"

if [[ ! -f "${WORK}/.target.done" ]]
then

    rm -rf "${WORK}/target"

    CUDA_VISIBLE_DEVICES=0 \
    python -m xmodel_kv.cli.extract \
        --model "${TARGET_MODEL}" \
        --tokens "${WORK}/tokens.npy" \
        --output "${WORK}/target" \
        --role target \
        --stride 4 \
        --device-map cuda:0 \
        --attn-implementation sdpa

    touch "${WORK}/.target.done"

else

    echo "Target extraction already completed."

fi


# ============================================================
# 4. Validate activation stores
# ============================================================

echo
echo "============================================================"
echo "[4] ACTIVATION STORE CHECK"
echo "============================================================"

python - \
    "${WORK}/source/metadata.json" \
    "${WORK}/target/metadata.json" <<'PY'

import json
import sys

for path in sys.argv[1:]:

    x = json.load(
        open(path)
    )

    print()
    print(path)

    print(
        "model =",
        x["model_path"],
    )

    print(
        "layers =",
        x["num_layers"],
    )

    print(
        "KV heads =",
        x["num_kv_heads"],
    )

    print(
        "head dim =",
        x["head_dim"],
    )

    print(
        "sequences =",
        x["num_sequences"],
    )

    print(
        "sequence length =",
        x["sequence_length"],
    )

    print(
        "stride =",
        x["stride"],
    )

    print(
        "observations =",
        x["num_observations"],
    )

    assert x["num_sequences"] == 125
    assert x["sequence_length"] == 4096
    assert x["stride"] == 4
    assert x["num_observations"] == 128000
PY

du -sh \
    "${WORK}/source" \
    "${WORK}/target"


# ============================================================
# 5. Source-layer selection
#
# Keep the same setting as old mapper.
# Selection uses 32,768 observations because selection.py
# preloads source activations as float32 onto the GPU.
# ============================================================

echo
echo "============================================================"
echo "[5] SOURCE-LAYER SELECTION"
echo "============================================================"

if [[ ! -f "${WORK}/.select.done" ]]
then

    rm -f \
        "${WORK}/selection_k12.json"

    CUDA_VISIBLE_DEVICES=0 \
    python -m xmodel_kv.cli.select \
        --source "${WORK}/source" \
        --target "${WORK}/target" \
        --output "${WORK}/selection_k12.json" \
        --top-k 12 \
        --ridge 0 \
        --device cuda:0 \
        --max-observations 32768

    touch "${WORK}/.select.done"

else

    echo "Selection already completed."

fi


python - \
    "${WORK}/selection_k12.json" <<'PY'

import json
import sys

x = json.load(
    open(sys.argv[1])
)

print(
    "selection observations =",
    x["num_observations"],
)

print(
    "top-k =",
    x["top_k"],
)

print(
    "target layers =",
    len(x["selected_layers"]),
)

assert x["num_observations"] == 32768
assert x["top_k"] == 12
PY


# ============================================================
# 6. Production mapper fit
#
# Use ALL 128,000 observations.
# ============================================================

echo
echo "============================================================"
echo "[6] PRODUCTION FIT"
echo "============================================================"

if [[ ! -f "${WORK}/.fit.done" ]]
then

    rm -rf \
        "${WORK}/mapper_k12"

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

    touch "${WORK}/.fit.done"

else

    echo "Mapper fit already completed."

fi


echo
echo "[Mapper config]"

cat \
    "${WORK}/mapper_k12/config.json"


echo
echo "[Mapper size]"

du -sh \
    "${WORK}/mapper_k12"


# ============================================================
# 7. Validate old vs new mapper controls
# ============================================================

echo
echo "============================================================"
echo "[7] OLD / NEW CONFIG CONTROL"
echo "============================================================"

python - <<'PY'
import json

OLD = (
    "work/crosskv_qwen3_17b_to_4b/"
    "mapper_k12/config.json"
)

NEW = (
    "work/crosskv_qwen3_17b_to_4b_longcal4k/"
    "mapper_k12/config.json"
)

old = json.load(
    open(OLD)
)

new = json.load(
    open(NEW)
)

fields = (
    "source_model",
    "target_model",
    "source_kv_heads",
    "target_kv_heads",
    "source_head_dim",
    "target_head_dim",
    "ridge",
    "weight_dtype",
    "num_observations",
)

for field in fields:

    print(
        field,
        "old=",
        old[field],
        "new=",
        new[field],
    )

    assert old[field] == new[field]


print(
    "old top-k =",
    len(old["selected_layers"][0]),
)

print(
    "new top-k =",
    len(new["selected_layers"][0]),
)

assert (
    len(old["selected_layers"][0])
    ==
    len(new["selected_layers"][0])
    ==
    12
)

print()
print(
    "CONTROL CHECK PASSED:"
    " same N/ridge/top-k/models;"
    " calibration context length differs."
)
PY


# ============================================================
# 8. Re-run THE SAME HotpotQA real handoff
# ============================================================

echo
echo "============================================================"
echo "[8] REAL HOTPOTQA HANDOFF WITH 4K-CAL MAPPER"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 \
python scripts/run_hotpot_crosskv_handoff.py \
    --source-model "${SOURCE_MODEL}" \
    --target-model "${TARGET_MODEL}" \
    --mapper \
        "${WORK}/mapper_k12" \
    --prompt-file \
        inputs/hotpotqa_index83_prompt.txt \
    --meta-file \
        inputs/hotpotqa_index83_meta.json \
    --output \
        "outputs/real_handoff_longcal4k/hotpot_index83_${SLURM_JOB_ID}.json" \
    --max-new-tokens 64


echo
echo "============================================================"
echo "[DONE]"
echo "============================================================"

echo "END=$(date --iso-8601=seconds)"

