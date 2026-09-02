#!/usr/bin/bash
#SBATCH --job-name=sparkv-lab-eval
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/sparkv-lab-eval-%j.out
#SBATCH --error=logs/sparkv-lab-eval-%j.err

set -Eeuo pipefail

readonly EXPECTED_BRANCH="exp/sparkv-test"
readonly SOURCE_JOB_ID="${SOURCE_JOB_ID:?Set SOURCE_JOB_ID to a preserved direct SparKV source job}"
readonly MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"

# Immediate lab-meeting smoke: 1 sample x 1 repeat.
# Recommended reported run: 4+ samples x 3 repeats after a 4-sample source job.
readonly EVAL_SAMPLES="${EVAL_SAMPLES:-1}"
readonly REPEATS="${REPEATS:-1}"
readonly QUALITY_TOKENS="${QUALITY_TOKENS:-32}"

readonly BANDWIDTH_MBPS="${BANDWIDTH_MBPS:-640}"
readonly JITTER_CV="${JITTER_CV:-0.0}"
readonly SEED="${SEED:-2026}"
readonly DELTA_MS="${DELTA_MS:-auto}"

readonly RUNTIME_WINDOW="${RUNTIME_WINDOW:-4}"
readonly IMBALANCE_MARGIN="${IMBALANCE_MARGIN:-0.05}"
readonly MAX_MIGRATIONS="${MAX_MIGRATIONS:-4}"

readonly SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

REPO_ROOT="$(
    git -C "${SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null
)" || {
    echo "[ERROR] Could not resolve repository root." >&2
    exit 2
}
readonly REPO_ROOT

readonly JOB_ID="${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"

readonly SOURCE_LOCAL_ROOT="/local_datasets/${USER}/sparkv-direct/${SOURCE_JOB_ID}"
readonly SOURCE_RESULT_ROOT="${REPO_ROOT}/results/sparkv/direct-${SOURCE_JOB_ID}"
readonly PREPARED="${SOURCE_LOCAL_ROOT}/prepared_triviaqa.pt"
readonly CLOUD_ROOT="${SOURCE_LOCAL_ROOT}/cloud"
readonly PROFILE_ROOT="${SOURCE_RESULT_ROOT}/scheduler_profiles"
readonly PREDICTOR="${SOURCE_RESULT_ROOT}/overhead_predictor.pt"

readonly RESULT_ROOT="${REPO_ROOT}/results/sparkv/lab-eval-${SOURCE_JOB_ID}-to-${JOB_ID}"
readonly SCHEDULE_ROOT="${RESULT_ROOT}/schedules"
readonly ALL_STREAM_ROOT="${RESULT_ROOT}/all_stream_schedules"
readonly RESULT_JSONL="${RESULT_ROOT}/evaluation.jsonl"
readonly SUMMARY_CSV="${RESULT_ROOT}/summary.csv"
readonly REPORT_MD="${RESULT_ROOT}/report.md"

cd "${REPO_ROOT}"

if [[ "$(git branch --show-current)" != "${EXPECTED_BRANCH}" ]]; then
    echo "[ERROR] Expected branch ${EXPECTED_BRANCH}; got $(git branch --show-current)." >&2
    exit 2
fi

if [[ "${ALLOW_DIRTY:-0}" != "1" ]] && [[ -n "$(git status --porcelain)" ]]; then
    echo "[ERROR] Refusing evaluation from a dirty worktree." >&2
    echo "[ERROR] Commit/stash intended changes, or set ALLOW_DIRTY=1 only for debugging." >&2
    git status --short >&2
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
    logs \
    "${RESULT_ROOT}" \
    "${SCHEDULE_ROOT}" \
    "${ALL_STREAM_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data/${USER}/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

for required in \
    "${PREPARED}" \
    "${PREDICTOR}"; do
    if [[ ! -f "${required}" ]]; then
        echo "[ERROR] Missing source artifact: ${required}" >&2
        exit 4
    fi
done

if [[ ! -d "${CLOUD_ROOT}" ]]; then
    echo "[ERROR] Missing cloud root: ${CLOUD_ROOT}" >&2
    exit 4
fi

for ((i=0; i<EVAL_SAMPLES; i++)); do
    sample="$(printf 'sample_%03d' "${i}")"
    if [[ ! -f "${PROFILE_ROOT}/${sample}.json" ]]; then
        echo "[ERROR] Missing scheduler profile for ${sample}." >&2
        echo "[ERROR] The source job was not built with EVAL_SAMPLES >= ${EVAL_SAMPLES}." >&2
        exit 4
    fi
    if [[ ! -f "${CLOUD_ROOT}/${sample}/meta.json" ]]; then
        echo "[ERROR] Missing cloud artifact for ${sample}." >&2
        exit 4
    fi
done

{
    echo "utc_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "job_id=${JOB_ID}"
    echo "source_job_id=${SOURCE_JOB_ID}"
    echo "hostname=$(hostname)"
    echo "branch=$(git branch --show-current)"
    echo "commit=$(git rev-parse HEAD)"
    echo "dirty=$(test -n "$(git status --porcelain)" && echo 1 || echo 0)"
    echo "model=${MODEL_ID}"
    echo "eval_samples=${EVAL_SAMPLES}"
    echo "repeats=${REPEATS}"
    echo "quality_tokens=${QUALITY_TOKENS}"
    echo "bandwidth_mbps=${BANDWIDTH_MBPS}"
    echo "jitter_cv=${JITTER_CV}"
    echo "seed=${SEED}"
    echo "delta_ms_spec=${DELTA_MS}"
    echo "runtime_window=${RUNTIME_WINDOW}"
    echo "imbalance_margin=${IMBALANCE_MARGIN}"
    echo "max_migrations=${MAX_MIGRATIONS}"
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
arch = f"sm{major}{minor}"
if arch not in {"sm80", "sm86", "sm87"}:
    raise SystemExit(f"Unsupported SpargeAttention Ampere path: {arch}")

print("torch:", torch.__version__)
print("bitarray:", getattr(bitarray, "__version__", "unknown"))
print("cuda_arch:", arch)
print("SpargeAttention/CUDA preflight: OK")
PY

python -m py_compile \
    baseline/sparkv/codec.py \
    baseline/sparkv/executor.py \
    baseline/sparkv/experiment.py \
    baseline/sparkv/evaluation.py \
    baseline/sparkv/overhead_model.py \
    baseline/sparkv/scheduler.py \
    baseline/sparkv/utils/summarize_lab_eval.py

python -m pytest -q baseline/sparkv/tests

python - "${PREDICTOR}" "${RESULT_ROOT}/predictor_metrics.json" <<'PY'
import json
import sys
import torch

payload = torch.load(
    sys.argv[1],
    map_location="cpu",
    weights_only=False,
)
metrics = payload.get("metrics", {})
Path = __import__("pathlib").Path
Path(sys.argv[2]).write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)
print(json.dumps(metrics, indent=2))
PY

echo "[1/3] Generate current-branch SparKV schedules"
for ((i=0; i<EVAL_SAMPLES; i++)); do
    sample="$(printf 'sample_%03d' "${i}")"
    python -m baseline.sparkv.scheduler \
        --profile "${PROFILE_ROOT}/${sample}.json" \
        --bandwidth-mbps "${BANDWIDTH_MBPS}" \
        --delta-ms "${DELTA_MS}" \
        --output "${SCHEDULE_ROOT}/${sample}.json"
done

echo "[2/3] Paired local / all-stream / SparKV evaluation"
python -m baseline.sparkv.evaluation \
    --model "${MODEL_ID}" \
    --prepared "${PREPARED}" \
    --cloud-root "${CLOUD_ROOT}" \
    --schedule-root "${SCHEDULE_ROOT}" \
    --all-stream-schedule-root "${ALL_STREAM_ROOT}" \
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

echo "[3/3] Summarize with interpretation guardrails"
python baseline/sparkv/utils/summarize_lab_eval.py \
    "${RESULT_JSONL}" \
    --csv "${SUMMARY_CSV}" \
    --report "${REPORT_MD}" \
    --predictor "${PREDICTOR}" \
    | tee "${RESULT_ROOT}/summary.txt"

echo "utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee -a "${RESULT_ROOT}/run_manifest.txt"

echo "[SUCCESS] Lab evaluation completed."
echo "[SUCCESS] Results: ${RESULT_ROOT}"
echo "[SUCCESS] Read first: ${REPORT_MD}"
