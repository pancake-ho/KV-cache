#!/bin/bash
#SBATCH --job-name=xkv-nd-sweep
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=10:00:00
#SBATCH --output=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.out
#SBATCH --error=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.err

set -euo pipefail

ROOT="/data/surt321/repos/lab/kv_cache/baseline/crosskv"

cd "${ROOT}"

source /data/surt321/anaconda3/etc/profile.d/conda.sh
conda activate lab

export HF_HOME=/data/surt321/hf_cache
export TOKENIZERS_PARALLELISM=false

SOURCE_MODEL="Qwen/Qwen3-0.6B"
TARGET_MODEL="Qwen/Qwen3-1.7B"

# ============================================================
# Existing Stage-2 activation store
# ============================================================

STAGE2_RUN="stage2_real_64x512_k4_143810"
BASE_WORK="${ROOT}/work/${STAGE2_RUN}"

SOURCE_DIR="${BASE_WORK}/source"
TARGET_DIR="${BASE_WORK}/target"

# Keep layer selection completely fixed.
FIXED_SELECTION="${BASE_WORK}/selection_n8192_k4.json"

# Existing Stage-4 result is supplied at submission time.
: "${STAGE4_RUN:?Submit with STAGE4_RUN=stage4_fixedsel_arc_<jobid>}"

STAGE4_OUT="${ROOT}/outputs/${STAGE4_RUN}"

ARC_ROOT="/data/surt321/datasets/arc_challenge"
ARC_FILE="${ARC_ROOT}/ARC-Challenge/test-00000-of-00001.parquet"

RUN_TAG="stage5_nd_ratio_arc_${SLURM_JOB_ID}"

WORK="${ROOT}/work/${RUN_TAG}"
OUT="${ROOT}/outputs/${RUN_TAG}"

mkdir -p "${WORK}"
mkdir -p "${OUT}"

echo "============================================================"
echo "CrossKV Stage 5"
echo "N/D transition sweep"
echo "============================================================"

echo "START=$(date --iso-8601=seconds)"
echo "JOB=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"

# ============================================================
# 0. Environment
# ============================================================

python - <<'PY'
import torch
import transformers
import tokenizers

print("torch        =", torch.__version__)
print("transformers =", transformers.__version__)
print("tokenizers   =", tokenizers.__version__)
print("CUDA         =", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: CUDA unavailable. CPU fallback is not allowed."
    )

print("GPU          =", torch.cuda.get_device_name(0))
print(
    "VRAM GiB     =",
    torch.cuda.get_device_properties(0).total_memory / 1024**3,
)
PY

nvidia-smi

echo
echo "[Git]"

git branch --show-current
git rev-parse HEAD
git status --short

# ============================================================
# 1. Validate inputs
# ============================================================

echo
echo "============================================================"
echo "[1] INPUT VALIDATION"
echo "============================================================"

for F in \
    "${SOURCE_DIR}/metadata.json" \
    "${TARGET_DIR}/metadata.json" \
    "${FIXED_SELECTION}" \
    "${ARC_FILE}" \
    "${STAGE4_OUT}/fixedsel_arc_sweep.json"
do

    if [[ ! -f "${F}" ]]; then

        echo "ERROR: required file missing:"
        echo "${F}"

        exit 1

    fi

done

python - "${SOURCE_DIR}" "${FIXED_SELECTION}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
selection_path = Path(sys.argv[2])

metadata = json.loads(
    (source / "metadata.json").read_text()
)

selection = json.loads(
    selection_path.read_text()
)

heads = int(metadata["num_kv_heads"])
head_dim = int(metadata["head_dim"])
top_k = int(selection["top_k"])

feature_dim = top_k * heads * head_dim

print("top_k       =", top_k)
print("kv_heads    =", heads)
print("head_dim    =", head_dim)
print("FEATURE D   =", feature_dim)

assert feature_dim == 4096
PY

# ============================================================
# 2. Provenance
# ============================================================

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job=${SLURM_JOB_ID}"
    echo "host=$(hostname)"

    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"

    echo "source_model=${SOURCE_MODEL}"
    echo "target_model=${TARGET_MODEL}"

    echo "stage2_run=${STAGE2_RUN}"
    echo "stage4_run=${STAGE4_RUN}"

    echo "fixed_selection=${FIXED_SELECTION}"

    echo "top_k=4"
    echo "feature_dimension=4096"
    echo "ridge=0.01"

    echo "new_N=1024,3072,4096,6144"
    echo "existing_N=512,2048,8192"

    echo "task=arc_challenge"

} > "${OUT}/provenance.txt"

python -m pip freeze \
    > "${OUT}/pip_freeze.txt"

nvidia-smi \
    > "${OUT}/nvidia_smi.txt"

# ============================================================
# 3. Fit new N values
#
# Existing:
#   512
#   2048
#   8192
#
# New:
#   1024
#   3072
#   4096
#   6144
# ============================================================

for N in 1024 3072 4096 6144
do

    echo
    echo "============================================================"
    echo "[2] FIT N=${N}"
    echo "============================================================"

    MAPPER="${WORK}/mapper_fixedsel_n${N}_k4"

    CUDA_VISIBLE_DEVICES=0 \
    python -m xmodel_kv.cli.fit \
        --source "${SOURCE_DIR}" \
        --target "${TARGET_DIR}" \
        --selection "${FIXED_SELECTION}" \
        --output "${MAPPER}" \
        --ridge 0.01 \
        --device cuda:0 \
        --max-observations "${N}" \
        --weight-dtype float32

    du -sh "${MAPPER}"

done

# ============================================================
# 4. Training R2 for new N
# ============================================================

echo
echo "============================================================"
echo "[3] TRAIN R2"
echo "============================================================"

python - "${WORK}" "${OUT}" <<'PY'
import json
import sys

from pathlib import Path

import numpy as np

from safetensors.torch import load_file

work = Path(sys.argv[1])
out = Path(sys.argv[2])

rows = []

for n in (1024, 3072, 4096, 6144):

    mapper = (
        work /
        f"mapper_fixedsel_n{n}_k4"
    )

    k = []
    v = []

    for file in sorted(
        (mapper / "layers").glob(
            "layer_*.safetensors"
        )
    ):

        x = load_file(file)

        k.append(
            x["k_r2"].float().mean().item()
        )

        v.append(
            x["v_r2"].float().mean().item()
        )

    row = {
        "observations": n,
        "n_over_d": n / 4096.0,
        "train_r2_k_mean":
            float(np.mean(k)),
        "train_r2_v_mean":
            float(np.mean(v)),
    }

    rows.append(row)

    print(row)

(out / "new_r2_summary.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
PY

# ============================================================
# 5. ARC evaluation
# ============================================================

for N in 1024 3072 4096 6144
do

    echo
    echo "============================================================"
    echo "[4] ARC-C N=${N}"
    echo "============================================================"

    MAPPER="${WORK}/mapper_fixedsel_n${N}_k4"

    RESULT="${OUT}/arc_n${N}.jsonl"

    CUDA_VISIBLE_DEVICES=0 \
    python -m xmodel_kv.cli.benchmark \
        --task arc_challenge \
        --dataset "${ARC_ROOT}" \
        --source-model "${SOURCE_MODEL}" \
        --target-model "${TARGET_MODEL}" \
        --mapper "${MAPPER}" \
        --source-device cuda:0 \
        --target-device cuda:0 \
        --output "${RESULT}"

    python -m xmodel_kv.cli.summarize_benchmark \
        --task arc_challenge \
        "${RESULT}" \
        --expected-documents 1172 \
        --output "${OUT}/arc_n${N}_summary.json"

done

# ============================================================
# 6. Combine existing Stage-4 + new Stage-5 values
# ============================================================

echo
echo "============================================================"
echo "[5] FULL N/D SWEEP"
echo "============================================================"

python - "${STAGE4_OUT}" "${OUT}" <<'PY'
import json
import sys

from pathlib import Path

stage4 = Path(sys.argv[1])
stage5 = Path(sys.argv[2])

existing = {
    int(row["observations"]): row
    for row in json.loads(
        (
            stage4 /
            "fixedsel_arc_sweep.json"
        ).read_text()
    )
}

rows = []

for n in (
    512,
    1024,
    2048,
    3072,
    4096,
    6144,
    8192,
):

    if n in existing:

        data = existing[n]

    else:

        data = json.loads(
            (
                stage5 /
                f"arc_n{n}_summary.json"
            ).read_text()
        )

    row = {
        "observations": n,
        "feature_dimension": 4096,
        "n_over_d": n / 4096.0,

        "standalone_acc_norm":
            data["standalone_acc_norm"],

        "transfer_acc_norm":
            data["transfer_acc_norm"],

        "accuracy_delta_pp":
            data["accuracy_delta_pp"],

        "retention_percent":
            data["retention_percent"],

        "floor_normalized_retention_percent":
            data[
                "floor_normalized_retention_percent"
            ],
    }

    rows.append(row)


print()
print(
    f"{'N':>8} "
    f"{'N/D':>8} "
    f"{'Native':>10} "
    f"{'Transfer':>10} "
    f"{'Delta(pp)':>12} "
    f"{'Retention':>12}"
)

for r in rows:

    print(
        f"{r['observations']:8d} "
        f"{r['n_over_d']:8.3f} "
        f"{r['standalone_acc_norm']*100:9.2f}% "
        f"{r['transfer_acc_norm']*100:9.2f}% "
        f"{r['accuracy_delta_pp']:12.2f} "
        f"{r['retention_percent']:11.2f}%"
    )

(stage5 / "full_nd_sweep.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
PY

# ============================================================
# 7. Create CSV for plotting
# ============================================================

python - "${OUT}" <<'PY'
import csv
import json
import sys

from pathlib import Path

out = Path(sys.argv[1])

rows = json.loads(
    (out / "full_nd_sweep.json").read_text()
)

with (
    out / "full_nd_sweep.csv"
).open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(rows)

print(
    (out / "full_nd_sweep.csv").resolve()
)
PY

# ============================================================
# 8. Final checks
# ============================================================

echo
echo "============================================================"
echo "[DONE]"
echo "============================================================"

find "${OUT}" \
    -maxdepth 1 \
    -type f \
    -print \
    | sort

du -sh "${WORK}"
du -sh "${OUT}"

echo "END=$(date --iso-8601=seconds)"
echo "OUTPUT=${OUT}"

