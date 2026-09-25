#!/bin/bash
#SBATCH --job-name=xkv-fixsel
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=08:00:00
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

# ------------------------------------------------------------
# Existing Stage-2 activation store
# ------------------------------------------------------------

STAGE2_RUN="stage2_real_64x512_k4_143810"
BASE_WORK="${ROOT}/work/${STAGE2_RUN}"

SOURCE_DIR="${BASE_WORK}/source"
TARGET_DIR="${BASE_WORK}/target"

# Critical control:
# Use ONE identical source-layer selection for every fit.
FIXED_SELECTION="${BASE_WORK}/selection_n8192_k4.json"

# ARC dataset
ARC_ROOT="/data/surt321/datasets/arc_challenge"
ARC_FILE="${ARC_ROOT}/ARC-Challenge/test-00000-of-00001.parquet"

# New outputs
RUN_TAG="stage4_fixedsel_arc_${SLURM_JOB_ID}"
WORK="${ROOT}/work/${RUN_TAG}"
OUT="${ROOT}/outputs/${RUN_TAG}"

mkdir -p "${WORK}"
mkdir -p "${OUT}"

echo "============================================================"
echo "CrossKV Stage 4"
echo "Fixed-selection calibration-size ablation"
echo "============================================================"
echo "START=$(date --iso-8601=seconds)"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "ROOT=${ROOT}"
echo "RUN_TAG=${RUN_TAG}"
echo

# ============================================================
# 0. Provenance / CUDA
# ============================================================

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
        "ERROR: CUDA unavailable. Aborting instead of CPU fallback."
    )

print("GPU          =", torch.cuda.get_device_name(0))
print(
    "VRAM GiB     =",
    torch.cuda.get_device_properties(0).total_memory / 1024**3,
)
PY

nvidia-smi

# ============================================================
# 1. Validate required existing artifacts
# ============================================================

echo
echo "============================================================"
echo "[1] INPUT VALIDATION"
echo "============================================================"

for PATH_TO_CHECK in \
    "${SOURCE_DIR}/metadata.json" \
    "${TARGET_DIR}/metadata.json" \
    "${FIXED_SELECTION}" \
    "${ARC_FILE}"
do
    if [[ ! -f "${PATH_TO_CHECK}" ]]; then
        echo "ERROR: Missing required file:"
        echo "${PATH_TO_CHECK}"
        exit 1
    fi
done

echo "SOURCE metadata:"
cat "${SOURCE_DIR}/metadata.json"

echo
echo "TARGET metadata:"
cat "${TARGET_DIR}/metadata.json"

echo
echo "Fixed selection:"
python - "${FIXED_SELECTION}" <<'PY'
import json
import sys

p = sys.argv[1]

with open(p) as f:
    x = json.load(f)

print("top_k            =", x["top_k"])
print("num_observations =", x["num_observations"])
print("target_layers    =", len(x["selected_layers"]))

print("first 10 selected layers:")
for i, layers in enumerate(x["selected_layers"][:10]):
    print(i, layers)
PY

# ARC row count
python - "${ARC_FILE}" <<'PY'
import sys
import pyarrow.parquet as pq

path = sys.argv[1]

metadata = pq.read_metadata(path)

print("ARC rows =", metadata.num_rows)

if metadata.num_rows != 1172:
    raise SystemExit(
        f"Unexpected ARC row count: {metadata.num_rows}"
    )
PY

# ============================================================
# 2. Record provenance
# ============================================================

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "source_model=${SOURCE_MODEL}"
    echo "target_model=${TARGET_MODEL}"
    echo "source_dir=${SOURCE_DIR}"
    echo "target_dir=${TARGET_DIR}"
    echo "fixed_selection=${FIXED_SELECTION}"
    echo "selection_based_on_N=8192"
    echo "fit_N=512,2048,8192"
    echo "top_k=4"
    echo "ridge=0.01"
    echo "task=arc_challenge"
} > "${OUT}/provenance.txt"

python -m pip freeze \
    > "${OUT}/pip_freeze.txt"

nvidia-smi \
    > "${OUT}/nvidia_smi.txt"

# ============================================================
# 3. Fit three mappers using IDENTICAL selected layers
# ============================================================

for N in 512 2048 8192
do

    echo
    echo "============================================================"
    echo "[2] FIXED-SELECTION FIT N=${N}"
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

    echo
    du -sh "${MAPPER}"

done

# ============================================================
# 4. Training R2 diagnostic
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

summary = []

for n in (512, 2048, 8192):

    root = work / f"mapper_fixedsel_n{n}_k4"

    k_r2 = []
    v_r2 = []

    for file in sorted(
        (root / "layers").glob("layer_*.safetensors")
    ):

        tensors = load_file(file)

        k_r2.append(
            tensors["k_r2"].float().mean().item()
        )

        v_r2.append(
            tensors["v_r2"].float().mean().item()
        )

    row = {
        "observations": n,
        "train_r2_k_mean": float(np.mean(k_r2)),
        "train_r2_v_mean": float(np.mean(v_r2)),
    }

    summary.append(row)
    print(row)

(out / "fixedsel_r2_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
PY

# ============================================================
# 5. ARC-Challenge evaluation
# ============================================================

for N in 512 2048 8192
do

    echo
    echo "============================================================"
    echo "[4] ARC EVALUATION FIXED-SELECTION N=${N}"
    echo "============================================================"

    MAPPER="${WORK}/mapper_fixedsel_n${N}_k4"

    RESULT="${OUT}/arc_fixedsel_n${N}.jsonl"

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
        --output "${OUT}/arc_fixedsel_n${N}_summary.json"

done

# ============================================================
# 6. Aggregate fixed-selection results
# ============================================================

echo
echo "============================================================"
echo "[5] FIXED-SELECTION ARC SUMMARY"
echo "============================================================"

python - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])

rows = []

for n in (512, 2048, 8192):

    data = json.loads(
        (
            out /
            f"arc_fixedsel_n{n}_summary.json"
        ).read_text()
    )

    rows.append({
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
    })

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

(out / "fixedsel_arc_sweep.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
PY

# ============================================================
# 7. Paired repair / break analysis
# ============================================================

echo
echo "============================================================"
echo "[6] PAIRED ANALYSIS"
echo "============================================================"

python - "${OUT}" <<'PY'
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])

results = {}

for n in (512, 2048, 8192):

    rows = {}

    with (
        out / f"arc_fixedsel_n{n}.jsonl"
    ).open() as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            rows[int(row["index"])] = row

    results[n] = rows


def exact_mcnemar_p(repaired, broken):

    total = repaired + broken

    if total == 0:
        return 1.0

    smaller = min(repaired, broken)

    tail = sum(
        math.comb(total, k)
        for k in range(smaller + 1)
    ) / (2 ** total)

    return min(1.0, 2.0 * tail)


for small in (512, 2048):

    repaired = 0
    broken = 0

    for idx in sorted(results[8192]):

        label = results[8192][idx]["label"]

        small_correct = (
            results[small][idx]
            ["transfer_prediction"]
            == label
        )

        large_correct = (
            results[8192][idx]
            ["transfer_prediction"]
            == label
        )

        if (not small_correct) and large_correct:
            repaired += 1

        if small_correct and (not large_correct):
            broken += 1

    p = exact_mcnemar_p(
        repaired,
        broken,
    )

    print(
        f"N={small} -> N=8192: "
        f"repaired={repaired}, "
        f"broken={broken}, "
        f"net={repaired-broken}, "
        f"exact_mcnemar_p={p:.8g}"
    )
PY

# ============================================================
# 8. Compare with Stage-3 free-selection results if available
# ============================================================

echo
echo "============================================================"
echo "[7] FREE-vs-FIXED SELECTION COMPARISON"
echo "============================================================"

STAGE3_OUT="${ROOT}/outputs/stage3_arc_143815"

python - "${STAGE3_OUT}" "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

free_root = Path(sys.argv[1])
fixed_root = Path(sys.argv[2])

if not (
    free_root / "arc_calibration_sweep.json"
).exists():

    print(
        "Stage-3 summary unavailable; "
        "skipping comparison."
    )

    raise SystemExit(0)

free = {
    int(x["observations"]): x
    for x in json.loads(
        (
            free_root /
            "arc_calibration_sweep.json"
        ).read_text()
    )
}

fixed = {
    int(x["observations"]): x
    for x in json.loads(
        (
            fixed_root /
            "fixedsel_arc_sweep.json"
        ).read_text()
    )
}

print()
print(
    f"{'N':>8} "
    f"{'FreeSel':>12} "
    f"{'FixedSel':>12} "
    f"{'Difference':>12}"
)

for n in (512, 2048, 8192):

    a = free[n]["transfer_acc_norm"]
    b = fixed[n]["transfer_acc_norm"]

    print(
        f"{n:8d} "
        f"{a*100:11.2f}% "
        f"{b*100:11.2f}% "
        f"{(b-a)*100:+11.2f}pp"
    )
PY

# ============================================================
# 9. Final output
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

