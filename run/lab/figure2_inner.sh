#!/usr/bin/env bash
# Called inside the pre-existing lab Docker container.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${ROOT}/run/common/runtime.sh"
mode="${1:?mode}"; dataset="${2:?dataset}"; tag="${3:?tag}"
scratch_root="${4:?scratch_root}"; result_dir="${5:?result_dir}"
kv_require_mode "${mode}"
kv_require_run_tag "${tag}"
[[ -f "${dataset}" ]] || { echo "ERROR: dataset not mounted inside container: ${dataset}" >&2; exit 2; }
[[ ! -e "${result_dir}" ]] || { echo "ERROR: result directory exists: ${result_dir}" >&2; exit 2; }
scratch="${scratch_root}/${tag}"
[[ ! -e "${scratch}" ]] || { echo "ERROR: scratch already exists: ${scratch}" >&2; exit 2; }
mkdir -p -- "${scratch}/work/provenance" "${result_dir}" "${scratch}/input"
if [[ "${KV_FIG2_DETACHED:-0}" == 1 ]]; then
  exec > "${result_dir}/docker.log" 2>&1
fi
trap 'code=$?; trap - EXIT; set +e; kv_export_small_results "${scratch}/work" "${result_dir}"; printf "status=%s\nexit_code=%s\nscratch=%s\n" "$([[ ${code} -eq 0 ]] && echo COMPLETE || echo FAILED)" "$code" "$scratch" > "${result_dir}/status.txt"; exit "$code"' EXIT
# Check scratch space BEFORE copying the input dataset.
[[ -x "${KV_PYTHON}" ]] || { echo "ERROR: Python not executable: ${KV_PYTHON}" >&2; exit 2; }
"${KV_PYTHON}" "${ROOT}/run/common/preflight.py" \
  --config "${ROOT}/baseline/crosskv/configs/figure2_qwen3_1p7b_4b.json" \
  --dataset "${dataset}" --scratch "${scratch}" --mode "${mode}" \
  --staging --output "${scratch}/work/provenance/preflight_staging.json"
# Copy single dataset file to scratch, if source is elsewhere.
local_dataset="${scratch}/input/$(basename -- "${dataset}")"
cp -p -- "${dataset}" "${local_dataset}"
export HF_HOME="${KV_FIG2_HF_HOME:-${scratch}/hf_cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export MPLCONFIGDIR="${scratch}/matplotlib"
mkdir -p -- "${HF_HOME}" "${MPLCONFIGDIR}"
kv_record_provenance "${scratch}/work/provenance/runtime.json" lab "${mode}"
nvidia-smi > "${scratch}/work/provenance/nvidia_smi.txt"
kv_run_figure2 "${mode}" "${local_dataset}" "${scratch}/work" "${result_dir}"
