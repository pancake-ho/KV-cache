#!/usr/bin/env bash
# Usage: bash jobs/figure2/submit.sh smoke|full [run_tag]
set -euo pipefail
FIG2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
mode="${1:?Usage: submit.sh smoke|full [run_tag]}"
[[ "${mode}" == smoke || "${mode}" == full ]] || { echo 'Mode must be smoke or full' >&2; exit 2; }
tag="${2:-figure2_${mode}_$(date +%Y%m%d_%H%M%S)}"
[[ "${tag}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo 'Use a simple run tag with letters, numbers, dots, underscores or hyphens.' >&2; exit 2; }
: "${FIG2_PARTITION:?Check sinfo and export FIG2_PARTITION}"
: "${FIG2_DATASET:?Export FIG2_DATASET to the local FineWeb-Edu file}"
[[ -f "${FIG2_DATASET}" ]] || { echo "Dataset missing: ${FIG2_DATASET}" >&2; exit 2; }
# Convert a relative dataset path before lib.sh changes the working directory.
FIG2_DATASET="$(cd -- "$(dirname -- "${FIG2_DATASET}")" && pwd)/$(basename -- "${FIG2_DATASET}")"
export FIG2_DATASET
REPO_ROOT="/data/surt321/repos/lab/kv_cache"
FIGURE2_DIR="${REPO_ROOT}/baseline/crosskv/jobs/figure2"
source "${FIGURE2_DIR}/lib.sh"
[[ -n "$(sinfo -h -p "${FIG2_PARTITION}" -o '%P')" ]] || { echo 'Partition not found' >&2; exit 2; }
export FIG2_MODE="${mode}"
export FIG2_RUN_DIR="${FIG2_ROOT}/work/${tag}"
mkdir -p "${FIG2_ROOT}/jobs/logs" "${FIG2_RUN_DIR}/provenance"
# This must succeed BEFORE any sbatch command. No GPU is required on the login node.
fig2_check_env --output "${FIG2_RUN_DIR}/provenance/environment_submit.json"
common=(--parsable --export=ALL --partition="${FIG2_PARTITION}" --gres=gpu:1
        --cpus-per-task="${FIG2_CPUS:-16}" --mem="${FIG2_MEM:-29G}"
        --time="${FIG2_TIME:-1-00:00:00}" --chdir="${FIG2_ROOT}"
        --output="${FIG2_ROOT}/jobs/logs/%x_%j.out" --error="${FIG2_ROOT}/jobs/logs/%x_%j.err")
first="$(sbatch "${common[@]}" --job-name=fig2-source "${FIG2_ROOT}/jobs/figure2/01_source.sh")"
first="${first%%;*}"
second="$(sbatch "${common[@]}" --dependency="afterok:${first}" --job-name=fig2-target "${FIG2_ROOT}/jobs/figure2/02_target.sh")"
second="${second%%;*}"
third="$(sbatch "${common[@]}" --dependency="afterok:${second}" --job-name=fig2-probe "${FIG2_ROOT}/jobs/figure2/03_probe_plot.sh")"
third="${third%%;*}"
printf 'Source job: %s\nTarget job: %s\nProbe/plot job: %s\nEnvironment: %s\nRun directory: %s\n' "${first}" "${second}" "${third}" "${FIG2_ENV}" "${FIG2_RUN_DIR}"
printf '%s\n' "${first}" "${second}" "${third}" > "${FIG2_RUN_DIR}/submitted_jobs.txt"
