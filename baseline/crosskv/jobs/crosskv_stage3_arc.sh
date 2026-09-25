#!/bin/bash
#SBATCH --job-name=xkv-arc
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=08:00:00
#SBATCH --output=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.out
#SBATCH --error=/data/surt321/repos/lab/kv_cache/baseline/crosskv/jobs/logs/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "[CrossKV Stage 3: ARC-Challenge downstream evaluation]"
echo "START=$(date --iso-8601=seconds)"
echo "JOB=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "============================================================"

# ============================================================
# 0. Environment
# ============================================================

source /data/surt321/anaconda3/etc/profile.d/conda.sh
conda activate lab

ROOT="/data/surt321/repos/lab/kv_cache/baseline/crosskv"
cd "${ROOT}"

export HF_HOME=/data/surt321/hf_cache
export TOKENIZERS_PARALLELISM=false

SOURCE_MODEL="Qwen/Qwen3-0.6B"
TARGET_MODEL="Qwen/Qwen3-1.7B"

STAGE2_RUN="stage2_real_64x512_k4_143810"
STAGE2_WORK="${ROOT}/work/${STAGE2_RUN}"

DATASET="/data/surt321/datasets/arc_challenge"

RUN_TAG="stage3_arc_${SLURM_JOB_ID}"
OUT="${ROOT}/outputs/${RUN_TAG}"

mkdir -p "${OUT}"

echo
echo "[Git]"
git branch --show-current
git rev-parse HEAD
git status --short

echo
echo "[Environment]"

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
        "ERROR: CUDA unavailable. "
        "Abort instead of silently using CPU."
    )

print("GPU          =", torch.cuda.get_device_name(0))
print(
    "VRAM GiB     =",
    torch.cuda.get_device_properties(0).total_memory / 1024**3,
)
PY

nvidia-smi

# ============================================================
# 1. Check existing Stage-2 artifacts
# ============================================================

echo
echo "============================================================"
echo "[1] CHECK STAGE-2 MAPPERS"
echo "============================================================"

for N in 512 2048 8192
do
    MAPPER="${STAGE2_WORK}/mapper_n${N}_k4"

    if [[ ! -f "${MAPPER}/config.json" ]]; then
        echo "ERROR: missing mapper: ${MAPPER}"
        exit 1
    fi

    echo
    echo "[Mapper N=${N}]"
    du -sh "${MAPPER}"

    python - "${MAPPER}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

config = json.loads(
    (root / "config.json").read_text()
)

print("source_model     =", config["source_model"])
print("target_model     =", config["target_model"])
print("num_observations =", config["num_observations"])
print("ridge            =", config["ridge"])
print("weight_dtype     =", config["weight_dtype"])
print("target_layers    =", config["target_layers"])
PY

done

# ============================================================
# 2. Check ARC-Challenge dataset
# ============================================================

echo
echo "============================================================"
echo "[2] CHECK ARC-CHALLENGE DATASET"
echo "============================================================"

ARC_FILE="${DATASET}/ARC-Challenge/test-00000-of-00001.parquet"

if [[ ! -f "${ARC_FILE}" ]]; then
    echo "ERROR: ARC-Challenge dataset not found:"
    echo "${ARC_FILE}"
    exit 1
fi

python - "${ARC_FILE}" <<'PY'
import sys
import pyarrow.parquet as pq

path = sys.argv[1]

metadata = pq.read_metadata(path)

print("ARC-Challenge rows =", metadata.num_rows)

if metadata.num_rows != 1172:
    raise SystemExit(
        f"Unexpected ARC-Challenge row count: "
        f"{metadata.num_rows}"
    )
PY

# ============================================================
# 3. Provenance
# ============================================================

{
    echo "date=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_branch=$(git branch --show-current)"
    echo "stage2_run=${STAGE2_RUN}"
    echo "source_model=${SOURCE_MODEL}"
    echo "target_model=${TARGET_MODEL}"
    echo "dataset=${DATASET}"
    echo "task=arc_challenge"
    echo "documents=1172"
    echo "mapper_top_k=4"
    echo "mapper_observations=512,2048,8192"
} > "${OUT}/provenance.txt"

python -m pip freeze \
    > "${OUT}/pip_freeze.txt"

nvidia-smi \
    > "${OUT}/nvidia_smi.txt"

# ============================================================
# 4. Full ARC-C evaluation for each calibration N
#
# IMPORTANT:
# Same source model
# Same target model
# Same ARC-C documents
# Same evaluator
#
# Only mapper calibration N differs.
# ============================================================

for N in 512 2048 8192
do

    echo
    echo "============================================================"
    echo "[3] ARC-C EVALUATION N=${N}"
    echo "============================================================"

    MAPPER="${STAGE2_WORK}/mapper_n${N}_k4"

    RESULT="${OUT}/arc_n${N}.jsonl"

    CUDA_VISIBLE_DEVICES=0 \
    python -m xmodel_kv.cli.benchmark \
        --task arc_challenge \
        --dataset "${DATASET}" \
        --source-model "${SOURCE_MODEL}" \
        --target-model "${TARGET_MODEL}" \
        --mapper "${MAPPER}" \
        --source-device cuda:0 \
        --target-device cuda:0 \
        --output "${RESULT}"

    echo
    echo "[Summary N=${N}]"

    python -m xmodel_kv.cli.summarize_benchmark \
        --task arc_challenge \
        "${RESULT}" \
        --expected-documents 1172 \
        --output "${OUT}/arc_n${N}_summary.json"

done

# ============================================================
# 5. Aggregate all mapper results
# ============================================================

echo
echo "============================================================"
echo "[4] AGGREGATED ARC-C RESULTS"
echo "============================================================"

python - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])

rows = []

for n in (512, 2048, 8192):

    summary_path = (
        out / f"arc_n{n}_summary.json"
    )

    data = json.loads(
        summary_path.read_text()
    )

    row = {
        "observations": n,
        "documents": data["documents"],
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
    f"{'Native':>10} "
    f"{'Transfer':>10} "
    f"{'Delta(pp)':>12} "
    f"{'Retention':>12} "
    f"{'FloorRet':>12}"
)

for row in rows:

    print(
        f"{row['observations']:8d} "
        f"{row['standalone_acc_norm']*100:9.2f}% "
        f"{row['transfer_acc_norm']*100:9.2f}% "
        f"{row['accuracy_delta_pp']:12.2f} "
        f"{row['retention_percent']:11.2f}% "
        f"{row['floor_normalized_retention_percent']:11.2f}%"
    )

(out / "arc_calibration_sweep.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
PY

# ============================================================
# 6. Pairwise prediction agreement diagnostics
# ============================================================

echo
echo "============================================================"
echo "[5] PREDICTION DIAGNOSTICS"
echo "============================================================"

python - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])

Ns = (512, 2048, 8192)

results = {}

for n in Ns:

    path = out / f"arc_n{n}.jsonl"

    rows = {}

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            rows[int(row["index"])] = row

    results[n] = rows

indices = sorted(
    set.intersection(
        *(set(results[n]) for n in Ns)
    )
)

print("common_documents =", len(indices))

# Native target prediction must be identical
# regardless of mapper N.
native_mismatch = 0

for idx in indices:

    preds = {
        results[n][idx]["standalone_prediction"]
        for n in Ns
    }

    if len(preds) != 1:
        native_mismatch += 1

print(
    "standalone_prediction_mismatch =",
    native_mismatch,
)

for a, b in (
    (512, 2048),
    (512, 8192),
    (2048, 8192),
):

    agree = sum(
        results[a][idx]["transfer_prediction"]
        ==
        results[b][idx]["transfer_prediction"]

        for idx in indices
    )

    print(
        f"transfer_agreement_{a}_{b} = "
        f"{agree / len(indices):.4f}"
    )

# Cases where increasing calibration to 8192
# repairs a smaller-N failure.
for small in (512, 2048):

    repaired = 0
    broken = 0

    for idx in indices:

        label = results[8192][idx]["label"]

        small_correct = (
            results[small][idx]["transfer_prediction"]
            == label
        )

        large_correct = (
            results[8192][idx]["transfer_prediction"]
            == label
        )

        if (not small_correct) and large_correct:
            repaired += 1

        if small_correct and (not large_correct):
            broken += 1

    print(
        f"N={small} -> N=8192 : "
        f"repaired={repaired}, "
        f"broken={broken}, "
        f"net={repaired-broken}"
    )
PY

# ============================================================
# 7. Disk / result checks
# ============================================================

echo
echo "============================================================"
echo "[6] FINAL OUTPUT CHECK"
echo "============================================================"

find "${OUT}" \
    -maxdepth 1 \
    -type f \
    -print \
    | sort

du -sh "${OUT}"

echo
echo "============================================================"
echo "[DONE]"
echo "END=$(date --iso-8601=seconds)"
echo "OUTPUT=${OUT}"
echo "============================================================"