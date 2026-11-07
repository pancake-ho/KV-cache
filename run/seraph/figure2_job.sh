#!/usr/bin/env bash
# Slurm compute-node script. Three logical phases run on the SAME allocated node.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${ROOT}/run/common/runtime.sh"
: "${SLURM_JOB_ID:?Must run inside Slurm allocation}"
: "${KV_FIG2_MODE:?}"
: "${KV_FIG2_DATASET:?}"
: "${KV_FIG2_TAG:?}"
: "${KV_FIG2_RESULT_DIR:?}"
: "${KV_FIG2_SCRATCH_ROOT:?}"
: "${KV_PYTHON:?}"
kv_require_mode "${KV_FIG2_MODE}"
kv_require_run_tag "${KV_FIG2_TAG}"
[[ "${KV_FIG2_SCRATCH_ROOT}" == /local_datasets/* ]] || exit 2
scratch="${KV_FIG2_SCRATCH_ROOT}/${KV_FIG2_TAG}_job${SLURM_JOB_ID}"
# Non-destructive: never remove scratch; unique job ID prevents clobbering runs.
[[ ! -e "${scratch}" ]] || { echo "ERROR: scratch already exists: ${scratch}" >&2; exit 2; }
mkdir -p -- "${scratch}/input" "${scratch}/work/provenance"
status="FAILED"
on_exit() {
  local code=$?
  trap - EXIT
  set +e
  kv_export_small_results "${scratch}/work" "${KV_FIG2_RESULT_DIR}"
  printf '%s\n' "status=$([[ ${code} -eq 0 ]] && echo COMPLETE || echo FAILED)" \
    "exit_code=${code}" "job_id=${SLURM_JOB_ID}" "scratch=${scratch}" \
    > "${KV_FIG2_RESULT_DIR}/status.txt"
  exit "${code}"
}
trap on_exit EXIT
# Check scratch space BEFORE copying the input dataset.
[[ -x "${KV_PYTHON}" ]] || { echo "ERROR: Python not executable: ${KV_PYTHON}" >&2; exit 2; }
"${KV_PYTHON}" "${ROOT}/run/common/preflight.py" \
  --config "${ROOT}/baseline/crosskv/configs/figure2_qwen3_1p7b_4b.json" \
  --dataset "${KV_FIG2_DATASET}" --scratch "${scratch}" --mode "${KV_FIG2_MODE}" \
  --staging --output "${scratch}/work/provenance/preflight_staging.json"
# Stage only a single dataset file; no direct NAS memmap training.
local_dataset="${scratch}/input/$(basename -- "${KV_FIG2_DATASET}")"
cp -p -- "${KV_FIG2_DATASET}" "${local_dataset}"
export HF_HOME="${KV_FIG2_HF_HOME:-${scratch}/hf_cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export MPLCONFIGDIR="${scratch}/matplotlib"
mkdir -p -- "${HF_HOME}" "${MPLCONFIGDIR}"
kv_record_provenance "${scratch}/work/provenance/runtime.json" seraph "${KV_FIG2_MODE}"
nvidia-smi > "${scratch}/work/provenance/nvidia_smi.txt"
kv_run_figure2 "${KV_FIG2_MODE}" "${local_dataset}" "${scratch}/work" "${KV_FIG2_RESULT_DIR}"
