#!/bin/bash
#SBATCH --job-name=xkv-heldout
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

DATASET="/data/surt321/datasets/fineweb-edu/sample/10BT/000_00000.parquet"

# ============================================================
# Existing runs
# ============================================================

STAGE2_RUN="stage2_real_64x512_k4_143810"

: "${STAGE4_RUN:?Submit with STAGE4_RUN=stage4_fixedsel_arc_<jobid>}"
: "${STAGE5_RUN:?Submit with STAGE5_RUN=stage5_nd_ratio_arc_<jobid>}"

STAGE2_WORK="${ROOT}/work/${STAGE2_RUN}"

STAGE4_WORK="${ROOT}/work/${STAGE4_RUN}"
STAGE4_OUT="${ROOT}/outputs/${STAGE4_RUN}"

STAGE5_WORK="${ROOT}/work/${STAGE5_RUN}"
STAGE5_OUT="${ROOT}/outputs/${STAGE5_RUN}"

RUN_TAG="stage6_heldout_r2_${SLURM_JOB_ID}"

WORK="${ROOT}/work/${RUN_TAG}"
OUT="${ROOT}/outputs/${RUN_TAG}"

mkdir -p "${WORK}"
mkdir -p "${OUT}"

echo "============================================================"
echo "CrossKV Stage 6"
echo "Held-out KV reconstruction generalization"
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
        "ERROR: CUDA unavailable. "
        "CPU fallback is not allowed."
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
# 1. Validate old runs
# ============================================================

for F in \
    "${STAGE2_WORK}/tokens.npy" \
    "${STAGE4_OUT}/fixedsel_r2_summary.json" \
    "${STAGE5_OUT}/new_r2_summary.json" \
    "${STAGE5_OUT}/full_nd_sweep.json"
do

    if [[ ! -f "${F}" ]]; then
        echo "ERROR: required file missing:"
        echo "${F}"
        exit 1
    fi

done

# ============================================================
# 2. Mapper path table
# ============================================================

declare -A MAPPERS

MAPPERS[512]="${STAGE4_WORK}/mapper_fixedsel_n512_k4"
MAPPERS[1024]="${STAGE5_WORK}/mapper_fixedsel_n1024_k4"
MAPPERS[2048]="${STAGE4_WORK}/mapper_fixedsel_n2048_k4"
MAPPERS[3072]="${STAGE5_WORK}/mapper_fixedsel_n3072_k4"
MAPPERS[4096]="${STAGE5_WORK}/mapper_fixedsel_n4096_k4"
MAPPERS[6144]="${STAGE5_WORK}/mapper_fixedsel_n6144_k4"
MAPPERS[8192]="${STAGE4_WORK}/mapper_fixedsel_n8192_k4"

for N in 512 1024 2048 3072 4096 6144 8192
do

    if [[ ! -f "${MAPPERS[$N]}/config.json" ]]; then

        echo "ERROR: missing mapper for N=${N}"
        echo "${MAPPERS[$N]}"

        exit 1

    fi

done

# ============================================================
# 3. Generate 128-sequence deterministic token set
#
# 128 * 512 / stride 4
# = 16384 observations
#
# First 64 sequences MUST exactly match old Stage-2 data.
# ============================================================

echo
echo "============================================================"
echo "[1] PREPARE 128 SEQUENCES"
echo "============================================================"

python -m xmodel_kv.cli.prepare \
    --dataset "${DATASET}" \
    --tokenizer "${SOURCE_MODEL}" \
    --output "${WORK}/tokens_128.npy" \
    --text-field text \
    --num-sequences 128 \
    --sequence-length 512

# ============================================================
# 4. Check exact calibration-prefix identity
# ============================================================

echo
echo "============================================================"
echo "[2] TOKEN PREFIX IDENTITY CHECK"
echo "============================================================"

python - \
    "${STAGE2_WORK}/tokens.npy" \
    "${WORK}/tokens_128.npy" <<'PY'

import sys
import numpy as np

old = np.load(sys.argv[1])
new = np.load(sys.argv[2])

print("old shape =", old.shape)
print("new shape =", new.shape)

assert old.shape == (64, 512)
assert new.shape == (128, 512)

same = np.array_equal(
    old,
    new[:64],
)

print(
    "first_64_sequences_identical =",
    same,
)

if not same:
    raise SystemExit(
        "ERROR: New token packing does not reproduce "
        "the original calibration prefix."
    )

print(
    "Held-out sequences =",
    64,
)

print(
    "Held-out raw tokens =",
    64 * 512,
)

print(
    "Held-out stride-4 observations =",
    64 * (512 // 4),
)

PY

# ============================================================
# 5. Extract 128-sequence source KV
# ============================================================

echo
echo "============================================================"
echo "[3] SOURCE KV EXTRACTION"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 \
python -m xmodel_kv.cli.extract \
    --model "${SOURCE_MODEL}" \
    --tokens "${WORK}/tokens_128.npy" \
    --output "${WORK}/source_128" \
    --role source \
    --stride 4 \
    --device-map cuda:0 \
    --attn-implementation sdpa

# ============================================================
# 6. Extract 128-sequence target KV
# ============================================================

echo
echo "============================================================"
echo "[4] TARGET KV EXTRACTION"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 \
python -m xmodel_kv.cli.extract \
    --model "${TARGET_MODEL}" \
    --tokens "${WORK}/tokens_128.npy" \
    --output "${WORK}/target_128" \
    --role target \
    --stride 4 \
    --device-map cuda:0 \
    --attn-implementation sdpa

cat "${WORK}/source_128/metadata.json"
cat "${WORK}/target_128/metadata.json"

# ============================================================
# 7. Held-out R²
#
# Calibration observations:
#   [0, 8192)
#
# Completely held-out observations:
#   [8192, 16384)
# ============================================================

echo
echo "============================================================"
echo "[5] HELD-OUT R2"
echo "============================================================"

python - \
    "${WORK}/source_128" \
    "${WORK}/target_128" \
    "${STAGE4_WORK}" \
    "${STAGE5_WORK}" \
    "${STAGE4_OUT}" \
    "${STAGE5_OUT}" \
    "${OUT}" <<'PY'

import json
import sys

from pathlib import Path

import numpy as np
import torch

from safetensors.torch import load_file

from xmodel_kv.artifact import MapperConfig
from xmodel_kv.store import ActivationStore


source_root = Path(sys.argv[1])
target_root = Path(sys.argv[2])

stage4_work = Path(sys.argv[3])
stage5_work = Path(sys.argv[4])

stage4_out = Path(sys.argv[5])
stage5_out = Path(sys.argv[6])

out = Path(sys.argv[7])


SOURCE = ActivationStore(source_root)
TARGET = ActivationStore(target_root)

assert SOURCE.metadata.num_observations == 16384
assert TARGET.metadata.num_observations == 16384

HOLDOUT_START = 8192
HOLDOUT_END = 16384

device = torch.device("cuda:0")


mapper_paths = {
    512:
        stage4_work /
        "mapper_fixedsel_n512_k4",

    1024:
        stage5_work /
        "mapper_fixedsel_n1024_k4",

    2048:
        stage4_work /
        "mapper_fixedsel_n2048_k4",

    3072:
        stage5_work /
        "mapper_fixedsel_n3072_k4",

    4096:
        stage5_work /
        "mapper_fixedsel_n4096_k4",

    6144:
        stage5_work /
        "mapper_fixedsel_n6144_k4",

    8192:
        stage4_work /
        "mapper_fixedsel_n8192_k4",
}


train_rows = {}

for file in (
    stage4_out / "fixedsel_r2_summary.json",
    stage5_out / "new_r2_summary.json",
):

    for row in json.loads(
        file.read_text()
    ):

        train_rows[
            int(row["observations"])
        ] = row


arc_rows = {
    int(row["observations"]): row
    for row in json.loads(
        (
            stage5_out /
            "full_nd_sweep.json"
        ).read_text()
    )
}


def heldout_r2_for_mapper(
    mapper_root: Path,
):

    config = MapperConfig.load(
        mapper_root
    )

    k_layer_scores = []
    v_layer_scores = []

    k_mse = []
    v_mse = []

    for target_layer, selected in enumerate(
        config.selected_layers
    ):

        tensors = load_file(
            mapper_root /
            "layers" /
            f"layer_{target_layer:03d}.safetensors",
            device="cpu",
        )

        for kind in ("k", "v"):

            x_parts = []

            for source_layer in selected:

                array = np.array(
                    SOURCE.open(
                        kind,
                        source_layer,
                    )[
                        HOLDOUT_START:
                        HOLDOUT_END
                    ],
                    dtype=np.float32,
                    copy=True,
                )

                x_parts.append(
                    torch.from_numpy(
                        array.reshape(
                            array.shape[0],
                            -1,
                        )
                    )
                )

            x = torch.cat(
                x_parts,
                dim=1,
            ).to(device)

            y_array = np.array(
                TARGET.open(
                    kind,
                    target_layer,
                )[
                    HOLDOUT_START:
                    HOLDOUT_END
                ],
                dtype=np.float32,
                copy=True,
            )

            y = torch.from_numpy(
                y_array.reshape(
                    y_array.shape[0],
                    -1,
                )
            ).to(device)

            weight = tensors[
                f"{kind}_weight"
            ].float().flatten(1).to(device)

            bias = tensors[
                f"{kind}_bias"
            ].float().flatten().to(device)

            prediction = (
                x @ weight
                + bias
            )

            mean_y = y.mean(
                dim=0
            )

            error = (
                y - prediction
            )

            sse = (
                error.square()
                .sum(dim=0)
            )

            sst = (
                (y - mean_y)
                .square()
                .sum(dim=0)
            )

            eps = torch.finfo(
                sst.dtype
            ).eps

            r2 = (
                1.0
                - sse /
                sst.clamp_min(eps)
            )

            mse = (
                error.square()
                .mean()
                .item()
            )

            score = (
                r2.mean()
                .item()
            )

            if kind == "k":
                k_layer_scores.append(
                    score
                )
                k_mse.append(mse)

            else:
                v_layer_scores.append(
                    score
                )
                v_mse.append(mse)

            del (
                x,
                y,
                weight,
                bias,
                prediction,
                error,
                sse,
                sst,
                r2,
                x_parts,
                y_array,
            )

            torch.cuda.empty_cache()

    return {
        "heldout_r2_k_mean":
            float(
                np.mean(
                    k_layer_scores
                )
            ),

        "heldout_r2_v_mean":
            float(
                np.mean(
                    v_layer_scores
                )
            ),

        "heldout_mse_k_mean":
            float(
                np.mean(k_mse)
            ),

        "heldout_mse_v_mean":
            float(
                np.mean(v_mse)
            ),
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

    print(
        f"Evaluating held-out mapper N={n}",
        flush=True,
    )

    heldout = heldout_r2_for_mapper(
        mapper_paths[n]
    )

    train = train_rows[n]
    arc = arc_rows[n]

    row = {
        "observations": n,
        "feature_dimension": 4096,
        "n_over_d": n / 4096.0,

        "train_r2_k_mean":
            train[
                "train_r2_k_mean"
            ],

        "train_r2_v_mean":
            train[
                "train_r2_v_mean"
            ],

        **heldout,

        "arc_transfer_acc":
            arc[
                "transfer_acc_norm"
            ],

        "arc_retention_percent":
            arc[
                "retention_percent"
            ],
    }

    rows.append(row)

    print(row)


(out / "heldout_r2_sweep.json").write_text(
    json.dumps(
        rows,
        indent=2,
    ) + "\n"
)


# ------------------------------------------------------------
# Descriptive correlations
# ------------------------------------------------------------

train_mean = np.asarray([
    (
        row["train_r2_k_mean"]
        +
        row["train_r2_v_mean"]
    ) / 2
    for row in rows
])

heldout_mean = np.asarray([
    (
        row["heldout_r2_k_mean"]
        +
        row["heldout_r2_v_mean"]
    ) / 2
    for row in rows
])

accuracy = np.asarray([
    row["arc_transfer_acc"]
    for row in rows
])

corr_train = float(
    np.corrcoef(
        train_mean,
        accuracy,
    )[0, 1]
)

corr_heldout = float(
    np.corrcoef(
        heldout_mean,
        accuracy,
    )[0, 1]
)

correlations = {
    "pearson_train_r2_vs_arc":
        corr_train,

    "pearson_heldout_r2_vs_arc":
        corr_heldout,

    "note":
        (
            "Descriptive only: "
            "the seven calibration settings "
            "are nested and not independent samples."
        ),
}

(
    out /
    "r2_arc_correlations.json"
).write_text(
    json.dumps(
        correlations,
        indent=2,
    ) + "\n"
)

print()
print("==============================================")
print("HELD-OUT SUMMARY")
print("==============================================")

print(
    f"{'N':>8} "
    f"{'N/D':>7} "
    f"{'TrainK':>9} "
    f"{'TrainV':>9} "
    f"{'HoldK':>9} "
    f"{'HoldV':>9} "
    f"{'ARC':>8}"
)

for r in rows:

    print(
        f"{r['observations']:8d} "
        f"{r['n_over_d']:7.3f} "
        f"{r['train_r2_k_mean']:9.4f} "
        f"{r['train_r2_v_mean']:9.4f} "
        f"{r['heldout_r2_k_mean']:9.4f} "
        f"{r['heldout_r2_v_mean']:9.4f} "
        f"{r['arc_transfer_acc']*100:7.2f}%"
    )

print()
print(
    "Pearson train R2 vs ARC =",
    corr_train,
)

print(
    "Pearson heldout R2 vs ARC =",
    corr_heldout,
)

PY

# ============================================================
# 8. Provenance
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
    echo "stage5_run=${STAGE5_RUN}"

    echo "total_sequences=128"
    echo "sequence_length=512"
    echo "stride=4"

    echo "calibration_observation_range=[0,8192)"
    echo "heldout_observation_range=[8192,16384)"

} > "${OUT}/provenance.txt"

python -m pip freeze \
    > "${OUT}/pip_freeze.txt"

nvidia-smi \
    > "${OUT}/nvidia_smi.txt"

# ============================================================
# 9. Final
# ============================================================

echo
echo "============================================================"
echo "[DONE]"
echo "============================================================"

cat "${OUT}/heldout_r2_sweep.json"

echo
cat "${OUT}/r2_arc_correlations.json"

echo
du -sh "${WORK}"
du -sh "${OUT}"

echo
echo "END=$(date --iso-8601=seconds)"
echo "OUTPUT=${OUT}"

