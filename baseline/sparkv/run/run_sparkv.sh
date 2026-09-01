#!/usr/bin/bash
#SBATCH --job-name=sparkv-direct
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/sparkv-direct-%j.out
#SBATCH --error=logs/sparkv-direct-%j.err

set -Eeuo pipefail

readonly EXPECTED_BRANCH="exp/sparkv-test"
readonly MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"

readonly EVAL_SAMPLES="${EVAL_SAMPLES:-1}"
readonly PROFILE_RECORDS="${PROFILE_RECORDS:-6000}"
readonly PREPARED_SAMPLES="${PREPARED_SAMPLES:-4}"

readonly PROMPT_TOKENS="${PROMPT_TOKENS:-8193}"
readonly CHUNK_SIZE="${CHUNK_SIZE:-1024}"

readonly REPEATS="${REPEATS:-1}"
readonly QUALITY_TOKENS="${QUALITY_TOKENS:-32}"

readonly BANDWIDTH_MBPS="${BANDWIDTH_MBPS:-640}"
readonly JITTER_CV="${JITTER_CV:-0.0}"
readonly SEED="${SEED:-2026}"

# Paper states 1024-token chunks, but does not disclose Delta t.
readonly DELTA_MS="${DELTA_MS:-5.0}"

# Motivation explicitly evaluates 5-bit + Huffman.
# Implementation states layer-wise non-uniform quantization but does not
# disclose the layer bit allocation.  Set a comma-separated layer bit plan
# here if you have the authors' allocation.
readonly LAYER_BITS="${LAYER_BITS:-5}"

# Paper specifies sliding-window adaptation and a per-stage migration limit,
# but does not disclose these numerical values.
readonly RUNTIME_WINDOW="${RUNTIME_WINDOW:-4}"
readonly IMBALANCE_MARGIN="${IMBALANCE_MARGIN:-0.05}"
readonly MAX_MIGRATIONS="${MAX_MIGRATIONS:-4}"

# Paper specifies MLP 48/24, 6000 samples, 80/20, SGD, MSE.
# LR / momentum / epoch count are not reported.
readonly PRED_EPOCHS="${PRED_EPOCHS:-400}"
readonly PRED_LR="${PRED_LR:-0.01}"
readonly PRED_MOMENTUM="${PRED_MOMENTUM:-0.9}"

readonly SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

REPO_ROOT="$(
    git -C "${SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null
)" || {
    echo "[ERROR] Could not resolve repository root." >&2
    exit 2
}
readonly REPO_ROOT

readonly JOB_ID="${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly LOCAL_ROOT="/local_datasets/${USER}/sparkv-direct/${JOB_ID}"

readonly PREPARED="${LOCAL_ROOT}/prepared_triviaqa.pt"
readonly CLOUD_ROOT="${LOCAL_ROOT}/cloud"

readonly RESULT_ROOT="${REPO_ROOT}/results/sparkv/direct-${JOB_ID}"
readonly OVERHEAD_JSONL="${RESULT_ROOT}/overhead_profile.jsonl"
readonly OVERHEAD_META="${RESULT_ROOT}/overhead_profile.meta.json"
readonly PREDICTOR="${RESULT_ROOT}/overhead_predictor.pt"
readonly STREAM_ROOT="${RESULT_ROOT}/stream_profiles"
readonly PROFILE_ROOT="${RESULT_ROOT}/scheduler_profiles"
readonly SCHEDULE_ROOT="${RESULT_ROOT}/schedules"
readonly RESULT_JSONL="${RESULT_ROOT}/sparkv.jsonl"

JOB_SUCCEEDED=0

on_error() {
    local code=$?
    echo "[ERROR] line=${BASH_LINENO[0]} exit=${code}" >&2
    echo "[ERROR] local workdir preserved: ${LOCAL_ROOT}" >&2
    exit "${code}"
}

cleanup() {
    if [[ "${JOB_SUCCEEDED}" -eq 1 && "${KEEP_LOCAL_WORKDIR:-0}" -eq 0 ]]; then
        case "${LOCAL_ROOT}" in
            /local_datasets/"${USER}"/sparkv-direct/*)
                rm -rf -- "${LOCAL_ROOT}"
                ;;
            *)
                echo "[WARN] refusing unexpected cleanup path: ${LOCAL_ROOT}" >&2
                ;;
        esac
    else
        echo "[INFO] local workdir preserved: ${LOCAL_ROOT}"
    fi
}

trap on_error ERR
trap cleanup EXIT

cd "${REPO_ROOT}"

if [[ "$(git branch --show-current)" != "${EXPECTED_BRANCH}" ]]; then
    echo "[ERROR] Expected branch ${EXPECTED_BRANCH}; got $(git branch --show-current)." >&2
    exit 2
fi

if (( (PROMPT_TOKENS - 1) % CHUNK_SIZE != 0 )); then
    echo "[ERROR] PROMPT_TOKENS-1 must be divisible by CHUNK_SIZE." >&2
    exit 2
fi

if (( PREPARED_SAMPLES < EVAL_SAMPLES )); then
    echo "[ERROR] PREPARED_SAMPLES must be >= EVAL_SAMPLES." >&2
    exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "lab" ]]; then
    CONDA_SH=""
    for candidate in \
        "/data/${USER}/anaconda3/etc/profile.d/conda.sh" \
        "/data/${USER}/miniconda3/etc/profile.d/conda.sh"; do
        if [[ -f "${candidate}" ]]; then
            CONDA_SH="${candidate}"
            break
        fi
    done

    if [[ -n "${CONDA_SH}" ]]; then
        # shellcheck disable=SC1090
        source "${CONDA_SH}"
        conda activate lab
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate lab
    else
        echo "[ERROR] Could not activate conda env lab." >&2
        exit 3
    fi
fi

mkdir -p \
    "${LOCAL_ROOT}/tmp" \
    "${CLOUD_ROOT}" \
    "${RESULT_ROOT}" \
    "${STREAM_ROOT}" \
    "${PROFILE_ROOT}" \
    "${SCHEDULE_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR="${LOCAL_ROOT}/tmp"
export HF_HOME="${HF_HOME:-/data/${USER}/hf_cache}"
export HF_DATASETS_CACHE="${LOCAL_ROOT}/hf_datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

{
    echo "utc_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "job_id=${JOB_ID}"
    echo "hostname=$(hostname)"
    echo "repo_root=${REPO_ROOT}"
    echo "branch=$(git branch --show-current)"
    echo "commit=$(git rev-parse HEAD)"
    echo "dirty=$(test -n "$(git status --porcelain)" && echo 1 || echo 0)"
    echo "python=$(command -v python)"
    echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"

    echo "model=${MODEL_ID}"
    echo "eval_samples=${EVAL_SAMPLES}"
    echo "prepared_samples=${PREPARED_SAMPLES}"
    echo "profile_records=${PROFILE_RECORDS}"
    echo "prompt_tokens=${PROMPT_TOKENS}"
    echo "chunk_size=${CHUNK_SIZE}"

    echo "bandwidth_mbps=${BANDWIDTH_MBPS}"
    echo "jitter_cv=${JITTER_CV}"
    echo "delta_ms=${DELTA_MS}"
    echo "layer_bits=${LAYER_BITS}"

    echo "runtime_window=${RUNTIME_WINDOW}"
    echo "imbalance_margin=${IMBALANCE_MARGIN}"
    echo "max_migrations=${MAX_MIGRATIONS}"

    echo "predictor_epochs=${PRED_EPOCHS}"
    echo "predictor_lr=${PRED_LR}"
    echo "predictor_momentum=${PRED_MOMENTUM}"
} | tee "${RESULT_ROOT}/run_manifest.txt"

git status --short > "${RESULT_ROOT}/git_status.txt"
python -m pip freeze > "${RESULT_ROOT}/pip_freeze.txt"

nvidia-smi \
    --query-gpu=name,memory.total,driver_version,compute_cap \
    --format=csv,noheader \
    | tee "${RESULT_ROOT}/gpu.txt"

python - <<'PY'
import torch
import bitarray
import spas_sage_attn

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")

major, minor = torch.cuda.get_device_capability(0)
if f"sm{major}{minor}" not in {"sm80", "sm86", "sm87"}:
    raise SystemExit(
        f"Direct SparKV wrapper expects Ampere sm80/sm86/sm87; got sm{major}{minor}"
    )

print("SpargeAttention/CUDA preflight: OK")
PY

python -m py_compile \
    baseline/sparkv/paper_codec.py \
    baseline/sparkv/paper_sparge.py \
    baseline/sparkv/overhead_model.py \
    baseline/sparkv/scheduler.py \
    baseline/sparkv/runtime_controller.py \
    baseline/sparkv/paper_artifacts.py \
    baseline/sparkv/paper_executor.py \
    baseline/sparkv/paper_experiment.py \
    baseline/sparkv/utils/validate_paper.py \
    baseline/sparkv/utils/summarize_paper.py

python -m pytest -q baseline/sparkv/tests

echo "[1/8] Prepare TriviaQA context"
python -m baseline.sparkv.experiment prepare \
    --model "${MODEL_ID}" \
    --samples "${PREPARED_SAMPLES}" \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --output "${PREPARED}"

echo "[2/8] Build actual compressed cloud KV artifacts"
python -m baseline.sparkv.paper_experiment build-cloud \
    --model "${MODEL_ID}" \
    --prepared "${PREPARED}" \
    --output-root "${CLOUD_ROOT}" \
    --samples "${EVAL_SAMPLES}" \
    --chunk-size "${CHUNK_SIZE}" \
    --layer-bits "${LAYER_BITS}" \
    --device cuda

du -sh "${CLOUD_ROOT}" \
    | tee "${RESULT_ROOT}/cloud_cache_size.txt"

echo "[3/8] Collect 6000 sparse-attention profiling records"
python -m baseline.sparkv.paper_experiment collect-overhead \
    --model "${MODEL_ID}" \
    --prepared "${PREPARED}" \
    --output "${OVERHEAD_JSONL}" \
    --chunk-size "${CHUNK_SIZE}" \
    --target-records "${PROFILE_RECORDS}" \
    --device cuda

echo "[4/8] Train SparKV 3-48-24-1 computation-latency MLP"
python -m baseline.sparkv.paper_experiment train-predictor \
    --profiles "${OVERHEAD_JSONL}" \
    --profile-meta "${OVERHEAD_META}" \
    --output "${PREDICTOR}" \
    --seed "${SEED}" \
    --epochs "${PRED_EPOCHS}" \
    --learning-rate "${PRED_LR}" \
    --momentum "${PRED_MOMENTUM}"

echo "[5/8] Build per-sample stream + scheduler profiles"
for ((i=0; i<EVAL_SAMPLES; i++)); do
    sample="$(printf 'sample_%03d' "${i}")"

    python -m baseline.sparkv.paper_experiment profile-stream \
        --sample-dir "${CLOUD_ROOT}/${sample}" \
        --output "${STREAM_ROOT}/${sample}.json" \
        --dtype bfloat16

    python -m baseline.sparkv.paper_experiment scheduler-profile \
        --model "${MODEL_ID}" \
        --prepared "${PREPARED}" \
        --sample-index "${i}" \
        --sample-dir "${CLOUD_ROOT}/${sample}" \
        --stream-profile "${STREAM_ROOT}/${sample}.json" \
        --predictor "${PREDICTOR}" \
        --output "${PROFILE_ROOT}/${sample}.json" \
        --chunk-size "${CHUNK_SIZE}" \
        --device cuda
done

echo "[6/8] Generate potential-aware SparKV schedules"
for ((i=0; i<EVAL_SAMPLES; i++)); do
    sample="$(printf 'sample_%03d' "${i}")"

    python -m baseline.sparkv.scheduler \
        --profile "${PROFILE_ROOT}/${sample}.json" \
        --bandwidth-mbps "${BANDWIDTH_MBPS}" \
        --delta-ms "${DELTA_MS}" \
        --output "${SCHEDULE_ROOT}/${sample}.json"
done

echo "[7/8] Direct SparKV execution"
python -m baseline.sparkv.paper_experiment run \
    --model "${MODEL_ID}" \
    --prepared "${PREPARED}" \
    --cloud-root "${CLOUD_ROOT}" \
    --schedule-root "${SCHEDULE_ROOT}" \
    --output "${RESULT_JSONL}" \
    --samples "${EVAL_SAMPLES}" \
    --repeats "${REPEATS}" \
    --quality-tokens "${QUALITY_TOKENS}" \
    --bandwidth-mbps "${BANDWIDTH_MBPS}" \
    --jitter-cv "${JITTER_CV}" \
    --seed "${SEED}" \
    --runtime-window "${RUNTIME_WINDOW}" \
    --imbalance-margin "${IMBALANCE_MARGIN}" \
    --max-migrations-per-stage "${MAX_MIGRATIONS}" \
    --device cuda

echo "[8/8] Validate and summarize"
python baseline/sparkv/utils/validate_paper.py \
    "${RESULT_JSONL}" \
    | tee "${RESULT_ROOT}/validation.txt"

python baseline/sparkv/utils/summarize_paper.py \
    "${RESULT_JSONL}" \
    --output "${RESULT_ROOT}/summary.csv" \
    | tee "${RESULT_ROOT}/summary.txt"

echo "utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee -a "${RESULT_ROOT}/run_manifest.txt"

echo "[SUCCESS] Direct SparKV pipeline completed."
echo "[SUCCESS] Results: ${RESULT_ROOT}"

JOB_SUCCEEDED=1
