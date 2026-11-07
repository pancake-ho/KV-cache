#!/bin/bash
#SBATCH --job-name=xkv-hs-full
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


SOURCE_MODEL="Qwen/Qwen3-0.6B"
TARGET_MODEL="Qwen/Qwen3-1.7B"


# ============================================================
# Required previous runs
# ============================================================

: "${STAGE4_RUN:?Set STAGE4_RUN}"
: "${STAGE5_RUN:?Set STAGE5_RUN}"
: "${STAGE7_RUN:?Set STAGE7_RUN}"


STAGE4_WORK="${ROOT}/work/${STAGE4_RUN}"
STAGE5_WORK="${ROOT}/work/${STAGE5_RUN}"
STAGE7_OUT="${ROOT}/outputs/${STAGE7_RUN}"


# ============================================================
# Only unresolved comparison:
#
# N=4096 : N/D = 1.0
# N=8192 : N/D = 2.0
#
# Same source-layer selection, top-k, lambda, model pair.
# ============================================================

declare -A MAPPERS

MAPPERS[4096]="${STAGE5_WORK}/mapper_fixedsel_n4096_k4"
MAPPERS[8192]="${STAGE4_WORK}/mapper_fixedsel_n8192_k4"


# ============================================================
# HellaSwag
# ============================================================

DATA_ROOT="/data/surt321/datasets/hellaswag"

DATASET=$(find "${DATA_ROOT}" \
    -type f \
    -iname "*validation*.parquet" \
    | head -1)


if [[ -z "${DATASET}" ]]; then

    echo "ERROR: HellaSwag validation parquet not found."

    exit 1

fi


EXPECTED_DOCS=10042


# Stable directory name is intentional.
#
# If Slurm times out, submit the same script again with the
# same FULL_RUN. Existing JSONL indices are skipped and the
# evaluation resumes.
FULL_RUN="${FULL_RUN:-stage8_hellaswag_full_4096_8192}"

OUT="${ROOT}/outputs/${FULL_RUN}"

mkdir -p "${OUT}"


echo "============================================================"
echo "CrossKV Stage 8"
echo "Full HellaSwag validation"
echo "N=4096 vs N=8192"
echo "============================================================"

echo "START=$(date --iso-8601=seconds)"
echo "JOB=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"

echo "FULL_RUN=${FULL_RUN}"
echo "DATASET=${DATASET}"
echo "EXPECTED_DOCS=${EXPECTED_DOCS}"


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
        "ERROR: CUDA unavailable. "
        "Abort instead of CPU fallback."
    )

print(
    "GPU          =",
    torch.cuda.get_device_name(0),
)

print(
    "VRAM GiB     =",
    torch.cuda.get_device_properties(
        0
    ).total_memory / 1024**3,
)
PY


nvidia-smi


echo
echo "[Git]"

git branch --show-current
git rev-parse HEAD
git status --short


# ============================================================
# 1. Validate dataset
# ============================================================

echo
echo "============================================================"
echo "[1] DATASET VALIDATION"
echo "============================================================"


python - "${DATASET}" <<'PY'
import sys

import pyarrow.parquet as pq


path = sys.argv[1]

meta = pq.read_metadata(path)

print("HellaSwag rows =", meta.num_rows)

if meta.num_rows != 10042:

    raise SystemExit(
        f"Unexpected HellaSwag row count: "
        f"{meta.num_rows}"
    )
PY


# ============================================================
# 2. Validate mappers
# ============================================================

echo
echo "============================================================"
echo "[2] MAPPER VALIDATION"
echo "============================================================"


for N in 4096 8192
do

    MAPPER="${MAPPERS[$N]}"

    if [[ ! -f "${MAPPER}/config.json" ]]; then

        echo "ERROR: mapper missing for N=${N}"
        echo "${MAPPER}"

        exit 1

    fi


    python - "${MAPPER}" "${N}" <<'PY'
import json
import sys

from pathlib import Path


root = Path(sys.argv[1])
expected_n = int(sys.argv[2])

cfg = json.loads(
    (root / "config.json").read_text()
)

print()
print("mapper =", root)
print("source =", cfg["source_model"])
print("target =", cfg["target_model"])
print("N      =", cfg["num_observations"])
print("ridge  =", cfg["ridge"])
print(
    "top-k  =",
    len(cfg["selected_layers"][0]),
)

assert cfg["num_observations"] == expected_n
assert abs(float(cfg["ridge"]) - 0.01) < 1e-12
assert len(cfg["selected_layers"][0]) == 4
PY

done


# ============================================================
# 3. Seed from Stage-7 first 2,000 documents
#
# Do this only once.
# Existing Stage-8 files are NEVER overwritten.
# ============================================================

echo
echo "============================================================"
echo "[3] SEED / RESUME CHECK"
echo "============================================================"


for N in 4096 8192
do

    SOURCE_JSONL="${STAGE7_OUT}/hellaswag_n${N}.jsonl"

    DEST_JSONL="${OUT}/hellaswag_n${N}.jsonl"


    if [[ ! -f "${SOURCE_JSONL}" ]]; then

        echo "ERROR: Stage-7 seed missing:"
        echo "${SOURCE_JSONL}"

        exit 1

    fi


    if [[ ! -f "${DEST_JSONL}" ]]; then

        echo "Initializing Stage-8 N=${N} from Stage-7 subset."

        cp \
            "${SOURCE_JSONL}" \
            "${DEST_JSONL}"

    else

        echo "Existing Stage-8 result found for N=${N}."
        echo "Will resume without overwriting."

    fi


    python - "${DEST_JSONL}" "${N}" <<'PY'
import json
import sys

from pathlib import Path


path = Path(sys.argv[1])
n = int(sys.argv[2])

indices = []

with path.open() as f:

    for line in f:

        if not line.strip():
            continue

        row = json.loads(line)

        indices.append(
            int(row["index"])
        )


if len(indices) != len(set(indices)):

    raise SystemExit(
        f"Duplicate indices found for N={n}"
    )


print(
    f"N={n}: completed documents = "
    f"{len(indices)}"
)

print(
    f"N={n}: min index = "
    f"{min(indices) if indices else None}"
)

print(
    f"N={n}: max index = "
    f"{max(indices) if indices else None}"
)


if len(indices) < 2000:

    raise SystemExit(
        f"N={n}: expected at least "
        f"the Stage-7 2000-document seed"
    )

PY

done


# ============================================================
# 4. Attempt-specific provenance
# ============================================================

ATTEMPT="${OUT}/attempt_${SLURM_JOB_ID}"

mkdir -p "${ATTEMPT}"


{
    echo "date=$(date --iso-8601=seconds)"
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"

    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"

    echo "source_model=${SOURCE_MODEL}"
    echo "target_model=${TARGET_MODEL}"

    echo "stage4_run=${STAGE4_RUN}"
    echo "stage5_run=${STAGE5_RUN}"
    echo "stage7_run=${STAGE7_RUN}"

    echo "full_run=${FULL_RUN}"

    echo "dataset=${DATASET}"
    echo "expected_documents=${EXPECTED_DOCS}"

    echo "N=4096,8192"
    echo "D=4096"
    echo "top_k=4"
    echo "ridge=0.01"

} > "${ATTEMPT}/provenance.txt"


python -m pip freeze \
    > "${ATTEMPT}/pip_freeze.txt"


nvidia-smi \
    > "${ATTEMPT}/nvidia_smi.txt"


# ============================================================
# 5. Full HellaSwag
#
# xkv hellaswag reads existing indices in the output JSONL.
# Already-completed Stage-7/Stage-8 rows are skipped.
# ============================================================

for N in 4096 8192
do

    echo
    echo "============================================================"
    echo "[4] FULL HELLASWAG N=${N}"
    echo "============================================================"


    MAPPER="${MAPPERS[$N]}"

    RESULT="${OUT}/hellaswag_n${N}.jsonl"


    CUDA_VISIBLE_DEVICES=0 \
    python -m xmodel_kv.cli.hellaswag \
        --dataset "${DATASET}" \
        --source-model "${SOURCE_MODEL}" \
        --target-model "${TARGET_MODEL}" \
        --mapper "${MAPPER}" \
        --source-device cuda:0 \
        --target-device cuda:0 \
        --output "${RESULT}"


    echo
    echo "[Coverage after N=${N}]"


    python - "${RESULT}" <<'PY'
import json
import sys


indices = []

with open(sys.argv[1]) as f:

    for line in f:

        if not line.strip():
            continue

        indices.append(
            int(json.loads(line)["index"])
        )


print("rows =", len(indices))
print("unique =", len(set(indices)))

if len(indices) != len(set(indices)):

    raise SystemExit(
        "Duplicate evaluation indices."
    )

PY


    python -m xmodel_kv.cli.summarize_hellaswag \
        "${RESULT}" \
        --expected-documents "${EXPECTED_DOCS}" \
        --output \
        "${OUT}/hellaswag_n${N}_summary.json"

done


# ============================================================
# 6. Aggregate full results
# ============================================================

echo
echo "============================================================"
echo "[5] FULL SUMMARY"
echo "============================================================"


python - "${OUT}" <<'PY'
import json
import sys

from pathlib import Path


out = Path(sys.argv[1])

rows = []


for n in (
    4096,
    8192,
):

    data = json.loads(
        (
            out /
            f"hellaswag_n{n}_summary.json"
        ).read_text()
    )


    rows.append({
        "observations":
            n,

        "n_over_d":
            n / 4096.0,

        "documents":
            data["documents"],

        "standalone_correct":
            data["standalone_correct"],

        "transfer_correct":
            data["transfer_correct"],

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
    f"{'N/D':>7} "
    f"{'Native':>10} "
    f"{'Transfer':>10} "
    f"{'Delta':>10} "
    f"{'Retention':>11} "
    f"{'FloorRet':>11}"
)


for row in rows:

    print(
        f"{row['observations']:8d} "
        f"{row['n_over_d']:7.3f} "
        f"{row['standalone_acc_norm']*100:9.2f}% "
        f"{row['transfer_acc_norm']*100:9.2f}% "
        f"{row['accuracy_delta_pp']:9.2f} "
        f"{row['retention_percent']:10.2f}% "
        f"{row['floor_normalized_retention_percent']:10.2f}%"
    )


(
    out /
    "hellaswag_full_summary.json"
).write_text(
    json.dumps(
        rows,
        indent=2,
    ) + "\n"
)

PY


# ============================================================
# 7. Full paired comparison
# ============================================================

echo
echo "============================================================"
echo "[6] FULL PAIRED ANALYSIS"
echo "============================================================"


python - "${OUT}" <<'PY'
import json
import math
import sys

from pathlib import Path


out = Path(sys.argv[1])


def load(n):

    result = {}

    with (
        out /
        f"hellaswag_n{n}.jsonl"
    ).open() as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            idx = int(row["index"])

            if idx in result:

                raise ValueError(
                    f"Duplicate index {idx} "
                    f"for N={n}"
                )

            result[idx] = row

    return result


a = load(4096)
b = load(8192)


common = sorted(
    set(a) & set(b)
)


if len(common) != 10042:

    raise SystemExit(
        f"Expected 10042 common docs, "
        f"found {len(common)}"
    )


native_mismatch = sum(
    a[i]["standalone_prediction"]
    !=
    b[i]["standalone_prediction"]

    for i in common
)


repaired = 0
broken = 0


for i in common:

    label = b[i]["label"]

    correct_4096 = (
        a[i]["transfer_prediction"]
        == label
    )

    correct_8192 = (
        b[i]["transfer_prediction"]
        == label
    )


    if (
        not correct_4096
        and correct_8192
    ):

        repaired += 1


    if (
        correct_4096
        and not correct_8192
    ):

        broken += 1


discordant = repaired + broken

difference = (
    repaired - broken
) / len(common)


if discordant:

    smaller = min(
        repaired,
        broken,
    )

    tail = sum(
        math.comb(
            discordant,
            k,
        )
        for k in range(
            smaller + 1
        )
    ) / (
        2 ** discordant
    )

    p_exact = min(
        1.0,
        2.0 * tail,
    )

else:

    p_exact = 1.0


# Approximate paired-difference standard error.
#
# Var(mean(Y-X)) =
# [b+c - (b-c)^2/n] / n^2
#
n = len(common)

variance = (
    discordant
    -
    (
        (repaired - broken) ** 2
        / n
    )
) / (
    n ** 2
)

se = math.sqrt(
    max(
        0.0,
        variance,
    )
)

ci_low = (
    difference
    - 1.96 * se
)

ci_high = (
    difference
    + 1.96 * se
)


result = {
    "documents":
        n,

    "standalone_prediction_mismatch":
        native_mismatch,

    "repaired_4096_to_8192":
        repaired,

    "broken_4096_to_8192":
        broken,

    "net_correct_gain":
        repaired - broken,

    "accuracy_difference":
        difference,

    "accuracy_difference_pp":
        100.0 * difference,

    "approx_95ci_difference_pp": [
        100.0 * ci_low,
        100.0 * ci_high,
    ],

    "exact_mcnemar_p":
        p_exact,
}


print(
    json.dumps(
        result,
        indent=2,
    )
)


(
    out /
    "paired_4096_vs_8192.json"
).write_text(
    json.dumps(
        result,
        indent=2,
    ) + "\n"
)

PY


# ============================================================
# 8. Final
# ============================================================

echo
echo "============================================================"
echo "[DONE]"
echo "============================================================"


cat \
    "${OUT}/hellaswag_full_summary.json"


echo


cat \
    "${OUT}/paired_4096_vs_8192.json"


echo


find "${OUT}" \
    -maxdepth 2 \
    -type f \
    -print \
    | sort


echo

du -sh "${OUT}"

echo

echo "END=$(date --iso-8601=seconds)"
echo "OUTPUT=${OUT}"

