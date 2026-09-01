#!/usr/bin/bash
#SBATCH --job-name=sparkv-resume
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/sparkv-resume-%j.out
#SBATCH --error=logs/sparkv-resume-%j.err

set -Eeuo pipefail

readonly EXPECTED_BRANCH="exp/sparkv-test"
readonly SOURCE_JOB_ID="${SOURCE_JOB_ID:?Set SOURCE_JOB_ID to a preserved SparKV job}"
readonly MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
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
REPO_ROOT="$(git -C "${SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null)" || {
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

readonly RESULT_ROOT="${REPO_ROOT}/results/sparkv/resume-${SOURCE_JOB_ID}-to-${JOB_ID}"
readonly SCHEDULE_ROOT="${RESULT_ROOT}/schedules"
readonly RESULT_JSONL="${RESULT_ROOT}/sparkv.jsonl"

cd "${REPO_ROOT}"

if [[ "$(git branch --show-current)" != "${EXPECTED_BRANCH}" ]]; then
    echo "[ERROR] Expected branch ${EXPECTED_BRANCH}; got $(git branch --show-current)." >&2
    exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "lab" ]]; then
    source "/data/${USER}/anaconda3/etc/profile.d/conda.sh"
    conda activate lab
fi

mkdir -p "logs" "${RESULT_ROOT}" "${SCHEDULE_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data/${USER}/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ ! -f "${PREPARED}" ]]; then
    echo "[ERROR] Missing prepared records: ${PREPARED}" >&2
    exit 3
fi
if [[ ! -d "${CLOUD_ROOT}" ]]; then
    echo "[ERROR] Missing cloud artifacts: ${CLOUD_ROOT}" >&2
    exit 3
fi

{
    echo "utc_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "resume_job_id=${JOB_ID}"
    echo "source_job_id=${SOURCE_JOB_ID}"
    echo "branch=$(git branch --show-current)"
    echo "commit=$(git rev-parse HEAD)"
    echo "dirty=$(test -n "$(git status --porcelain)" && echo 1 || echo 0)"
    echo "delta_ms_spec=${DELTA_MS}"
    echo "bandwidth_mbps=${BANDWIDTH_MBPS}"
} | tee "${RESULT_ROOT}/run_manifest.txt"

python -m pytest -q baseline/sparkv/tests

echo "[6/8] Generate potential-aware SparKV schedules"
for ((i=0; i<EVAL_SAMPLES; i++)); do
    sample="$(printf 'sample_%03d' "${i}")"
    profile="${PROFILE_ROOT}/${sample}.json"

    if [[ ! -f "${profile}" ]]; then
        echo "[ERROR] Missing scheduler profile: ${profile}" >&2
        exit 4
    fi

    python -m baseline.sparkv.scheduler \
        --profile "${profile}" \
        --bandwidth-mbps "${BANDWIDTH_MBPS}" \
        --delta-ms "${DELTA_MS}" \
        --output "${SCHEDULE_ROOT}/${sample}.json"
done

echo "[7/8] Direct SparKV execution"
python -m baseline.sparkv.experiment run \
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
python baseline/sparkv/utils/validate_results.py \
    "${RESULT_JSONL}" \
    | tee "${RESULT_ROOT}/validation.txt"

python baseline/sparkv/utils/summarize_results.py \
    "${RESULT_JSONL}" \
    --output "${RESULT_ROOT}/summary.csv" \
    | tee "${RESULT_ROOT}/summary.txt"

echo "utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee -a "${RESULT_ROOT}/run_manifest.txt"

echo "[SUCCESS] SparKV resume completed."
echo "[SUCCESS] Results: ${RESULT_ROOT}"
