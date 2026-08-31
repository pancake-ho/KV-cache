#!/usr/bin/bash
#SBATCH --job-name=sparkv-cpu-smoke
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=29G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/sparkv-cpu-smoke-%j.out
#SBATCH --error=logs/sparkv-cpu-smoke-%j.err

set -Eeuo pipefail

readonly EXPECTED_BRANCH="exp/sparkv-test"
readonly MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
readonly CPU_DTYPE="${CPU_DTYPE:-float32}"
readonly SAMPLES="${SAMPLES:-1}"
readonly PROMPT_TOKENS="${PROMPT_TOKENS:-1025}"
readonly CHUNK_SIZE="${CHUNK_SIZE:-256}"
readonly REPEATS="${REPEATS:-1}"
readonly QUALITY_TOKENS="${QUALITY_TOKENS:-4}"
readonly BANDWIDTH_MBPS="${BANDWIDTH_MBPS:-640}"
readonly JITTER_CV="${JITTER_CV:-0.0}"
readonly SPLIT="${SPLIT:-2}"
readonly SEED="${SEED:-2026}"

# Directory from which sbatch was submitted.
readonly SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# Resolve the KV-cache repository root regardless of the submission
# location inside the repository.
REPO_ROOT="$(git -C "${SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "[ERROR] Could not locate the Git repository root from:" >&2
    echo "[ERROR]   ${SUBMIT_DIR}" >&2
    echo "[ERROR] Submit this job from somewhere inside the KV-cache repository." >&2
    exit 2
}
readonly REPO_ROOT

readonly JOB_ID="${SLURM_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly LOCAL_ROOT="/local_datasets/${USER}/sparkv-cpu-smoke/${JOB_ID}"
readonly CACHE_ROOT="${LOCAL_ROOT}/cache"
readonly PREPARED="${LOCAL_ROOT}/prepared_triviaqa.pt"
readonly RESULT_ROOT="${REPO_ROOT}/results/sparkv/cpu-smoke-${JOB_ID}"
readonly PROFILE_JSON="${RESULT_ROOT}/profile.json"

JOB_SUCCEEDED=0

on_error() {
    local exit_code=$?
    echo "[ERROR] line=${BASH_LINENO[0]} exit_code=${exit_code}" >&2
    echo "[ERROR] Temporary files are preserved at: ${LOCAL_ROOT}" >&2
    exit "${exit_code}"
}

cleanup() {
    if [[ "${JOB_SUCCEEDED}" -eq 1 && "${KEEP_LOCAL_WORKDIR:-0}" -eq 0 ]]; then
        case "${LOCAL_ROOT}" in
            /local_datasets/"${USER}"/sparkv-cpu-smoke/*)
                rm -rf -- "${LOCAL_ROOT}"
                echo "[INFO] Removed successful job's temporary cache: ${LOCAL_ROOT}"
                ;;
            *)
                echo "[WARN] Refusing to remove unexpected path: ${LOCAL_ROOT}" >&2
                ;;
        esac
    else
        echo "[INFO] Temporary files preserved at: ${LOCAL_ROOT}"
    fi
}

trap on_error ERR
trap cleanup EXIT

cd "${REPO_ROOT}"

if (( (PROMPT_TOKENS - 1) % CHUNK_SIZE != 0 )); then
    echo "[ERROR] PROMPT_TOKENS - 1 must be divisible by CHUNK_SIZE." >&2
    exit 2
fi

readonly NUM_CHUNKS=$(( (PROMPT_TOKENS - 1) / CHUNK_SIZE ))
if (( SPLIT < 0 || SPLIT > NUM_CHUNKS )); then
    echo "[ERROR] SPLIT must be between 0 and NUM_CHUNKS (${NUM_CHUNKS})." >&2
    exit 2
fi

if [[ "${CPU_DTYPE}" != "float32" && "${CPU_DTYPE}" != "bfloat16" ]]; then
    echo "[ERROR] CPU_DTYPE must be float32 or bfloat16." >&2
    exit 2
fi

if [[ ! -f baseline/sparkv/experiment.py || ! -f baseline/sparkv/scheduler.py ]]; then
    echo "[ERROR] Invalid KV-cache repository layout." >&2
    echo "[ERROR] Detected repository root: ${REPO_ROOT}" >&2
    echo "[ERROR] Expected:" >&2
    echo "[ERROR]   ${REPO_ROOT}/baseline/sparkv/experiment.py" >&2
    echo "[ERROR]   ${REPO_ROOT}/baseline/sparkv/scheduler.py" >&2
    exit 2
fi

if [[ "$(git branch --show-current)" != "${EXPECTED_BRANCH}" ]]; then
    echo "[ERROR] Expected branch '${EXPECTED_BRANCH}', found '$(git branch --show-current)'." >&2
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
        echo "[ERROR] Conda environment 'lab' is not active and Conda was not found." >&2
        echo "[ERROR] Run 'conda activate lab' before sbatch submission." >&2
        exit 3
    fi
fi

mkdir -p "${LOCAL_ROOT}/tmp" "${CACHE_ROOT}" "${RESULT_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR="${LOCAL_ROOT}/tmp"
export HF_HOME="${HF_HOME:-/data/${USER}/hf_cache}"
export HF_DATASETS_CACHE="${LOCAL_ROOT}/hf_datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUDA_VISIBLE_DEVICES=""

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

{
    echo "utc_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "job_id=${JOB_ID}"
    echo "hostname=$(hostname)"
    echo "repo_root=${REPO_ROOT}"
    echo "branch=$(git branch --show-current)"
    echo "commit=$(git rev-parse HEAD)"
    echo "python=$(command -v python)"
    echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
    echo "requested_device=cpu"
    echo "cpu_dtype=${CPU_DTYPE}"
    echo "model=${MODEL_ID}"
    echo "samples=${SAMPLES}"
    echo "prompt_tokens=${PROMPT_TOKENS}"
    echo "chunk_size=${CHUNK_SIZE}"
    echo "repeats=${REPEATS}"
    echo "quality_tokens=${QUALITY_TOKENS}"
    echo "bandwidth_mbps=${BANDWIDTH_MBPS}"
    echo "jitter_cv=${JITTER_CV}"
    echo "split=${SPLIT}"
    echo "seed=${SEED}"
} | tee "${RESULT_ROOT}/run_manifest.txt"

git status --short > "${RESULT_ROOT}/git_status.txt"
python -m pip freeze > "${RESULT_ROOT}/pip_freeze.txt"

python - <<'PY' | tee "${RESULT_ROOT}/preflight.txt"
import json
from importlib.metadata import version

import torch

packages = [
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "safetensors",
    "numpy",
    "pandas",
    "psutil",
    "nvidia-ml-py",
    "pytest",
]

installed = {}
for package in packages:
    try:
        installed[package] = version(package)
    except Exception as exc:
        raise SystemExit(f"Required package is unavailable: {package}: {exc}") from exc

if torch.cuda.is_available():
    raise SystemExit(
        "CPU job unexpectedly exposes CUDA. Check CUDA_VISIBLE_DEVICES and Slurm directives."
    )

print(json.dumps({
    "packages": installed,
    "cuda_available": torch.cuda.is_available(),
    "cuda_visible_devices": "",
    "cpu_threads": torch.get_num_threads(),
}, indent=2))
PY

if ! python -m pip check | tee "${RESULT_ROOT}/pip_check.txt"; then
    echo "[WARN] pip reported an environment conflict; continuing to import/tests." >&2
fi

python -m py_compile \
    baseline/sparkv/experiment.py \
    baseline/sparkv/scheduler.py \
    baseline/sparkv/utils/validate_results.py \
    baseline/sparkv/utils/summarize_results.py

python -m pytest -q baseline/sparkv/tests \
    | tee "${RESULT_ROOT}/pytest.txt"

echo "[STEP 1/7] Preparing one LongBench/TriviaQA sample"
python -m baseline.sparkv.experiment prepare \
    --model "${MODEL_ID}" \
    --samples "${SAMPLES}" \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --output "${PREPARED}"

echo "[STEP 2/7] Building raw and q5 KV-cache files on CPU"
python -m baseline.sparkv.experiment build-cache \
    --model "${MODEL_ID}" \
    --prepared "${PREPARED}" \
    --cache-root "${CACHE_ROOT}" \
    --formats raw q5 \
    --samples "${SAMPLES}" \
    --chunk-size "${CHUNK_SIZE}" \
    --device cpu \
    --cpu-dtype "${CPU_DTYPE}"

du -sh "${CACHE_ROOT}" | tee "${RESULT_ROOT}/cache_size.txt"
cp "${CACHE_ROOT}/raw/sample_000/meta.json" "${RESULT_ROOT}/raw_meta.json"
cp "${CACHE_ROOT}/q5/sample_000/meta.json" "${RESULT_ROOT}/q5_meta.json"

echo "[STEP 3/7] Profiling per-chunk, per-layer CPU time"
python -m baseline.sparkv.experiment profile \
    --model "${MODEL_ID}" \
    --prepared "${PREPARED}" \
    --samples "${SAMPLES}" \
    --chunk-size "${CHUNK_SIZE}" \
    --output "${PROFILE_JSON}" \
    --device cpu \
    --cpu-dtype "${CPU_DTYPE}"

echo "[STEP 4/7] Generating offline scheduler outputs"
for format in raw q5; do
    python -m baseline.sparkv.scheduler \
        --profile "${PROFILE_JSON}" \
        --cache-meta "${CACHE_ROOT}/${format}/sample_000/meta.json" \
        --bandwidth-mbps "${BANDWIDTH_MBPS}" \
        --processing-ms 0.02 \
        --delta-ms 5.0 \
        --output "${RESULT_ROOT}/schedule_${format}.json"
done

echo "[STEP 5/7] Running raw-cache strategies on CPU"
python -m baseline.sparkv.experiment run \
    --model "${MODEL_ID}" \
    --prepared "${PREPARED}" \
    --cache-root "${CACHE_ROOT}" \
    --format raw \
    --strategies local fetch static adaptive \
    --split "${SPLIT}" \
    --bandwidth-mbps "${BANDWIDTH_MBPS}" \
    --jitter-cv "${JITTER_CV}" \
    --samples "${SAMPLES}" \
    --repeats "${REPEATS}" \
    --quality-tokens "${QUALITY_TOKENS}" \
    --seed "${SEED}" \
    --output "${RESULT_ROOT}/raw.jsonl" \
    --device cpu \
    --cpu-dtype "${CPU_DTYPE}"

echo "[STEP 6/7] Running q5-cache strategies on CPU"
python -m baseline.sparkv.experiment run \
    --model "${MODEL_ID}" \
    --prepared "${PREPARED}" \
    --cache-root "${CACHE_ROOT}" \
    --format q5 \
    --strategies local fetch static adaptive \
    --split "${SPLIT}" \
    --bandwidth-mbps "${BANDWIDTH_MBPS}" \
    --jitter-cv "${JITTER_CV}" \
    --samples "${SAMPLES}" \
    --repeats "${REPEATS}" \
    --quality-tokens "${QUALITY_TOKENS}" \
    --seed "${SEED}" \
    --output "${RESULT_ROOT}/q5.jsonl" \
    --device cpu \
    --cpu-dtype "${CPU_DTYPE}"

echo "[STEP 7/7] Validating raw equivalence and summarizing metrics"
python baseline/sparkv/utils/validate_results.py "${RESULT_ROOT}/raw.jsonl" \
    | tee "${RESULT_ROOT}/validation.txt"

python baseline/sparkv/utils/summarize_results.py \
    "${RESULT_ROOT}/raw.jsonl" \
    "${RESULT_ROOT}/q5.jsonl" \
    --output "${RESULT_ROOT}/summary.csv" \
    | tee "${RESULT_ROOT}/summary.txt"

echo "utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee -a "${RESULT_ROOT}/run_manifest.txt"

echo "[SUCCESS] SparKV CPU smoke pipeline completed."
echo "[SUCCESS] Persistent results: ${RESULT_ROOT}"
echo "[NOTICE] CPU timing is a functional check and must not be compared with GPU TTFT."

JOB_SUCCEEDED=1
