#!/bin/bash
#SBATCH --job-name=xkv-hswag
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=12:00:00
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
# Existing runs
# ============================================================

: "${STAGE4_RUN:?Set STAGE4_RUN}"
: "${STAGE5_RUN:?Set STAGE5_RUN}"


STAGE4_WORK="${ROOT}/work/${STAGE4_RUN}"
STAGE5_WORK="${ROOT}/work/${STAGE5_RUN}"


# ============================================================
# Representative mapper conditions
#
# N=512  : N/D = 0.125
# N=4096 : N/D = 1.0
# N=8192 : N/D = 2.0
#
# All use the SAME N=8192 source-layer selection.
# ============================================================

declare -A MAPPERS

MAPPERS[512]="${STAGE4_WORK}/mapper_fixedsel_n512_k4"

MAPPERS[4096]="${STAGE5_WORK}/mapper_fixedsel_n4096_k4"

MAPPERS[8192]="${STAGE4_WORK}/mapper_fixedsel_n8192_k4"


# ============================================================
# HellaSwag
# ============================================================

DATA_ROOT="/data/surt321/datasets/hellaswag"

MAX_DOCS=2000


DATASET=$(find "${DATA_ROOT}" \
    -type f \
    -iname "*validation*.parquet" \
    | head -1)


if [[ -z "${DATASET}" ]]; then

    echo "ERROR: HellaSwag validation parquet not found."

    exit 1

fi


RUN_TAG="stage7_hellaswag_subset_${SLURM_JOB_ID}"

OUT="${ROOT}/outputs/${RUN_TAG}"

mkdir -p "${OUT}"


echo "============================================================"
echo "CrossKV Stage 7"
echo "HellaSwag cross-task validation"
echo "============================================================"

echo "START=$(date --iso-8601=seconds)"
echo "JOB=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"

echo "DATASET=${DATASET}"
echo "MAX_DOCS=${MAX_DOCS}"


# ============================================================
# 0. Environment
# ============================================================

python - <<'PY'
import torch
import transformers
import tokenizers

print(
    "torch        =",
    torch.__version__,
)

print(
    "transformers =",
    transformers.__version__,
)

print(
    "tokenizers   =",
    tokenizers.__version__,
)

print(
    "CUDA         =",
    torch.cuda.is_available(),
)

if not torch.cuda.is_available():

    raise SystemExit(
        "ERROR: CUDA unavailable. "
        "CPU fallback is forbidden."
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

python - "${DATASET}" <<'PY'
import sys

import pyarrow.parquet as pq


path = sys.argv[1]

metadata = pq.read_metadata(path)

print(
    "HellaSwag validation rows =",
    metadata.num_rows,
)

if metadata.num_rows != 10042:

    raise SystemExit(
        f"Unexpected HellaSwag row count: "
        f"{metadata.num_rows}"
    )
PY


# ============================================================
# 2. Validate mappers
# ============================================================

for N in 512 4096 8192
do

    MAPPER="${MAPPERS[$N]}"

    if [[ ! -f "${MAPPER}/config.json" ]]; then

        echo "ERROR: mapper missing for N=${N}"

        echo "${MAPPER}"

        exit 1

    fi

    echo
    echo "[Mapper N=${N}]"

    python - "${MAPPER}" <<'PY'
import json
import sys

from pathlib import Path


root = Path(sys.argv[1])

config = json.loads(
    (root / "config.json").read_text()
)

print(
    "source_model =",
    config["source_model"],
)

print(
    "target_model =",
    config["target_model"],
)

print(
    "N            =",
    config["num_observations"],
)

print(
    "ridge        =",
    config["ridge"],
)

print(
    "top_k        =",
    len(config["selected_layers"][0]),
)
PY

done


# ============================================================
# 3. Provenance
# ============================================================

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job=${SLURM_JOB_ID}"
    echo "host=$(hostname)"

    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"

    echo "source_model=${SOURCE_MODEL}"
    echo "target_model=${TARGET_MODEL}"

    echo "stage4_run=${STAGE4_RUN}"
    echo "stage5_run=${STAGE5_RUN}"

    echo "dataset=${DATASET}"
    echo "max_documents=${MAX_DOCS}"

    echo "N=512,4096,8192"
    echo "feature_dimension=4096"
    echo "ridge=0.01"
    echo "top_k=4"

} > "${OUT}/provenance.txt"


if [[ -f "${DATA_ROOT}/REVISION.txt" ]]; then

    cp \
        "${DATA_ROOT}/REVISION.txt" \
        "${OUT}/dataset_revision.txt"

fi


python -m pip freeze \
    > "${OUT}/pip_freeze.txt"


nvidia-smi \
    > "${OUT}/nvidia_smi.txt"


# ============================================================
# 4. HellaSwag evaluation
# ============================================================

for N in 512 4096 8192
do

    echo
    echo "============================================================"
    echo "[HellaSwag N=${N}]"
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
        --max-documents "${MAX_DOCS}" \
        --output "${RESULT}"


    python -m xmodel_kv.cli.summarize_hellaswag \
        "${RESULT}" \
        --expected-documents "${MAX_DOCS}" \
        --output \
        "${OUT}/hellaswag_n${N}_summary.json"

done


# ============================================================
# 5. Aggregate
# ============================================================

echo
echo "============================================================"
echo "[SUMMARY]"
echo "============================================================"


python - "${OUT}" <<'PY'
import json
import sys

from pathlib import Path


out = Path(sys.argv[1])

rows = []


for n in (
    512,
    4096,
    8192,
):

    summary = json.loads(
        (
            out /
            f"hellaswag_n{n}_summary.json"
        ).read_text()
    )

    rows.append({
        "observations": n,

        "n_over_d":
            n / 4096.0,

        "documents":
            summary["documents"],

        "standalone_correct":
            summary["standalone_correct"],

        "transfer_correct":
            summary["transfer_correct"],

        "standalone_acc_norm":
            summary["standalone_acc_norm"],

        "transfer_acc_norm":
            summary["transfer_acc_norm"],

        "accuracy_delta_pp":
            summary["accuracy_delta_pp"],

        "retention_percent":
            summary["retention_percent"],

        "floor_normalized_retention_percent":
            summary[
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
    "hellaswag_calibration_sweep.json"
).write_text(
    json.dumps(
        rows,
        indent=2,
    ) + "\n"
)
PY


# ============================================================
# 6. Paired analysis
# ============================================================

echo
echo "============================================================"
echo "[PAIRED ANALYSIS]"
echo "============================================================"


python - "${OUT}" <<'PY'
import json
import math
import sys

from pathlib import Path


out = Path(sys.argv[1])


results = {}


for n in (
    512,
    4096,
    8192,
):

    rows = {}

    with (
        out /
        f"hellaswag_n{n}.jsonl"
    ).open() as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            rows[
                int(row["index"])
            ] = row

    results[n] = rows


common = sorted(
    set.intersection(
        *(
            set(results[n])
            for n in results
        )
    )
)


print(
    "common_documents =",
    len(common),
)


# Native predictions must be identical.
native_mismatch = 0


for idx in common:

    predictions = {
        results[n][idx][
            "standalone_prediction"
        ]
        for n in results
    }

    if len(predictions) != 1:
        native_mismatch += 1


print(
    "standalone_prediction_mismatch =",
    native_mismatch,
)


def exact_mcnemar(
    repaired,
    broken,
):

    total = repaired + broken

    if total == 0:
        return 1.0

    small = min(
        repaired,
        broken,
    )

    tail = sum(
        math.comb(total, k)
        for k in range(
            small + 1
        )
    ) / (2 ** total)

    return min(
        1.0,
        2.0 * tail,
    )


for small in (
    512,
    4096,
):

    repaired = 0
    broken = 0


    for idx in common:

        label = results[8192][idx][
            "label"
        ]

        small_correct = (
            results[small][idx][
                "transfer_prediction"
            ]
            == label
        )

        large_correct = (
            results[8192][idx][
                "transfer_prediction"
            ]
            == label
        )

        if (
            not small_correct
            and large_correct
        ):
            repaired += 1

        if (
            small_correct
            and not large_correct
        ):
            broken += 1


    p = exact_mcnemar(
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
# 7. Final
# ============================================================

echo
echo "============================================================"
echo "[DONE]"
echo "============================================================"

cat \
    "${OUT}/hellaswag_calibration_sweep.json"

echo

find "${OUT}" \
    -maxdepth 1 \
    -type f \
    -print \
    | sort

echo

du -sh "${OUT}"

echo

echo "END=$(date --iso-8601=seconds)"
echo "OUTPUT=${OUT}"

