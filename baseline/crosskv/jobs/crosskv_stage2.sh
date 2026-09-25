#!/bin/bash
#SBATCH --job-name=xkv-stage2
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=04:00:00
#SBATCH --output=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.out
#SBATCH --error=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.err

set -euo pipefail

echo "=================================================="
echo "[START] $(date)"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "=================================================="

# --------------------------------------------------
# 0. Environment
# --------------------------------------------------

source /data/surt321/anaconda3/etc/profile.d/conda.sh
conda activate lab

cd /data/surt321/repos/lab/kv_cache/baseline/crosskv

export HF_HOME=/data/surt321/hf_cache
export TOKENIZERS_PARALLELISM=false

echo "[Git]"
git branch --show-current
git rev-parse HEAD
git status --short

echo "[Environment]"
python - <<'PY'
import torch
import transformers
import tokenizers

print("torch       =", torch.__version__)
print("transformers=", transformers.__version__)
print("tokenizers  =", tokenizers.__version__)
print("CUDA        =", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable. Abort instead of CPU fallback.")

print("GPU         =", torch.cuda.get_device_name(0))
print(
    "VRAM GiB    =",
    torch.cuda.get_device_properties(0).total_memory / 1024**3,
)
PY

nvidia-smi

# --------------------------------------------------
# 1. Core unit tests
# --------------------------------------------------

echo "=================================================="
echo "[1] CORE TESTS"
echo "=================================================="

python -m pytest -q \
    tests/test_ridge.py \
    tests/test_rope.py \
    tests/test_tiny_qwen_cache.py

# --------------------------------------------------
# 2. Experiment configuration
# --------------------------------------------------

SOURCE_MODEL="Qwen/Qwen3-0.6B"
TARGET_MODEL="Qwen/Qwen3-1.7B"

DATASET="/data/surt321/datasets/fineweb-edu/sample/10BT/000_00000.parquet"

RUN_TAG="stage2_real_64x512_k4_${SLURM_JOB_ID}"

WORK="work/${RUN_TAG}"
OUT="outputs/${RUN_TAG}"

mkdir -p "${WORK}"
mkdir -p "${OUT}"

NUM_SEQ=64
SEQ_LEN=512
STRIDE=4
TOPK=4
RIDGE=0.01

echo "RUN_TAG=${RUN_TAG}"
echo "WORK=${WORK}"
echo "OUT=${OUT}"

# Provenance
{
    echo "date=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_branch=$(git branch --show-current)"
    echo "source_model=${SOURCE_MODEL}"
    echo "target_model=${TARGET_MODEL}"
    echo "dataset=${DATASET}"
    echo "num_sequences=${NUM_SEQ}"
    echo "sequence_length=${SEQ_LEN}"
    echo "stride=${STRIDE}"
    echo "top_k=${TOPK}"
    echo "ridge=${RIDGE}"
} > "${OUT}/provenance.txt"

python -m pip freeze > "${OUT}/pip_freeze.txt"
nvidia-smi > "${OUT}/nvidia_smi.txt"

# --------------------------------------------------
# 3. Prepare fixed calibration tokens
# --------------------------------------------------

echo "=================================================="
echo "[2] PREPARE TOKENS"
echo "=================================================="

python -m xmodel_kv.cli.prepare \
    --dataset "${DATASET}" \
    --tokenizer "${SOURCE_MODEL}" \
    --output "${WORK}/tokens.npy" \
    --text-field text \
    --num-sequences "${NUM_SEQ}" \
    --sequence-length "${SEQ_LEN}"

# --------------------------------------------------
# 4. Source KV extraction
# --------------------------------------------------

echo "=================================================="
echo "[3] SOURCE EXTRACTION"
echo "=================================================="

CUDA_VISIBLE_DEVICES=0 \
python -m xmodel_kv.cli.extract \
    --model "${SOURCE_MODEL}" \
    --tokens "${WORK}/tokens.npy" \
    --output "${WORK}/source" \
    --role source \
    --stride "${STRIDE}" \
    --device-map cuda:0 \
    --attn-implementation sdpa

# --------------------------------------------------
# 5. Target KV extraction
# --------------------------------------------------

echo "=================================================="
echo "[4] TARGET EXTRACTION"
echo "=================================================="

CUDA_VISIBLE_DEVICES=0 \
python -m xmodel_kv.cli.extract \
    --model "${TARGET_MODEL}" \
    --tokens "${WORK}/tokens.npy" \
    --output "${WORK}/target" \
    --role target \
    --stride "${STRIDE}" \
    --device-map cuda:0 \
    --attn-implementation sdpa

echo "[Activation metadata]"

cat "${WORK}/source/metadata.json"
cat "${WORK}/target/metadata.json"

du -sh "${WORK}"

# --------------------------------------------------
# 6. Calibration-size sweep
#
# Full extraction:
# 64 seq × 512 tokens / stride 4
# = 8192 observations
# --------------------------------------------------

for N in 512 2048 8192
do
    echo "=================================================="
    echo "[5] SELECTION N=${N}"
    echo "=================================================="

    CUDA_VISIBLE_DEVICES=0 \
    python -m xmodel_kv.cli.select \
        --source "${WORK}/source" \
        --target "${WORK}/target" \
        --output "${WORK}/selection_n${N}_k${TOPK}.json" \
        --top-k "${TOPK}" \
        --ridge 0 \
        --device cuda:0 \
        --max-observations "${N}"

    echo "=================================================="
    echo "[6] FIT N=${N}"
    echo "=================================================="

    CUDA_VISIBLE_DEVICES=0 \
    python -m xmodel_kv.cli.fit \
        --source "${WORK}/source" \
        --target "${WORK}/target" \
        --selection "${WORK}/selection_n${N}_k${TOPK}.json" \
        --output "${WORK}/mapper_n${N}_k${TOPK}" \
        --ridge "${RIDGE}" \
        --device cuda:0 \
        --max-observations "${N}" \
        --weight-dtype float32

    du -sh "${WORK}/mapper_n${N}_k${TOPK}"
done

# --------------------------------------------------
# 7. R2 diagnostics
# --------------------------------------------------

echo "=================================================="
echo "[7] R2 SUMMARY"
echo "=================================================="

python - "${WORK}" "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

work = Path(sys.argv[1])
out = Path(sys.argv[2])

rows = []

for n in (512, 2048, 8192):
    selection_path = work / f"selection_n{n}_k4.json"
    mapper_dir = work / f"mapper_n{n}_k4"

    selection = json.loads(selection_path.read_text())
    scores = np.asarray(selection["scores"])
    selected = selection["selected_layers"]

    selected_scores = []
    for target_layer, source_layers in enumerate(selected):
        selected_scores.extend(
            scores[target_layer, source_layer]
            for source_layer in source_layers
        )

    k_values = []
    v_values = []

    for layer_file in sorted((mapper_dir / "layers").glob("layer_*.safetensors")):
        tensors = load_file(layer_file)
        k_values.append(tensors["k_r2"].float().mean().item())
        v_values.append(tensors["v_r2"].float().mean().item())

    row = {
        "observations": n,
        "selection_selected_score_mean": float(np.mean(selected_scores)),
        "train_r2_k_mean": float(np.mean(k_values)),
        "train_r2_v_mean": float(np.mean(v_values)),
    }
    rows.append(row)

    print(row)

(out / "r2_summary.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
PY

# --------------------------------------------------
# 8. Same held-out continuation for all mappers
# --------------------------------------------------

echo "=================================================="
echo "[8] CONTINUATION VERIFY"
echo "=================================================="

VERIFY_TEXT="Large language models process long contexts during autoregressive inference. Each transformer layer produces key and value representations that are cached so previously processed tokens do not need to be recomputed. When an inference system switches from a smaller language model to a larger related model, the cached tensors are normally incompatible, forcing the larger model to prefill the entire history again. Cross-model KV cache transfer attempts to transform the smaller model's cache into the representation space expected by the larger model. If the transformation preserves the information used by attention, the receiving model can continue generation without repeating the complete prefill operation. This experiment evaluates how the quality of that transferred cache changes as more calibration observations are used to estimate the linear mapping."

for N in 512 2048 8192
do
    echo "--------------------------------------------------"
    echo "VERIFY N=${N}"
    echo "--------------------------------------------------"

    CUDA_VISIBLE_DEVICES=0 \
    python -m xmodel_kv.cli.verify \
        --source-model "${SOURCE_MODEL}" \
        --target-model "${TARGET_MODEL}" \
        --mapper "${WORK}/mapper_n${N}_k${TOPK}" \
        --source-device cuda:0 \
        --target-device cuda:0 \
        --prefix-length 64 \
        --max-length 256 \
        --text "${VERIFY_TEXT}" \
        | tee "${OUT}/verify_n${N}.json"
done

# --------------------------------------------------
# 9. Aggregate verify results
# --------------------------------------------------

echo "=================================================="
echo "[9] VERIFY SUMMARY"
echo "=================================================="

python - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])

rows = []

for n in (512, 2048, 8192):
    path = out / f"verify_n{n}.json"
    data = json.loads(path.read_text())

    row = {
        "observations": n,
        **data,
    }
    rows.append(row)

print(
    f"{'N':>8} "
    f"{'standalone':>12} "
    f"{'transfer':>12} "
    f"{'delta':>10} "
    f"{'top1':>10} "
    f"{'lp_mae':>10}"
)

for r in rows:
    print(
        f"{r['observations']:8d} "
        f"{r['standalone_nll']:12.4f} "
        f"{r['transfer_nll']:12.4f} "
        f"{r['nll_delta']:10.4f} "
        f"{r['top1_agreement']:10.4f} "
        f"{r['logprob_mae']:10.4f}"
    )

(out / "verify_summary.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
PY

# --------------------------------------------------
# 10. Final checks
# --------------------------------------------------

echo "=================================================="
echo "[10] OUTPUTS"
echo "=================================================="

du -sh "${WORK}"
du -sh "${OUT}"

find "${OUT}" -maxdepth 1 -type f -print | sort

echo "=================================================="
echo "[DONE] $(date)"
echo "=================================================="