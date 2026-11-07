#!/usr/bin/env bash
# Shared, platform-neutral environment checks and Figure 2 execution.
set -euo pipefail

kv_repo_root() {
  local current
  current="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
  printf '%s\n' "${current}"
}

kv_require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: command not found: $1" >&2; return 2; }
}

kv_require_run_tag() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "ERROR: invalid run tag: $1" >&2; return 2;
  }
}

kv_require_mode() {
  [[ "$1" == smoke || "$1" == full ]] || {
    echo "ERROR: mode must be smoke or full" >&2; return 2;
  }
}

kv_setup_python() {
  : "${KV_PYTHON:?Specify KV_PYTHON to the existing Python interpreter}"
  [[ -x "${KV_PYTHON}" ]] || { echo "ERROR: Python not executable: ${KV_PYTHON}" >&2; return 2; }
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
  local root
  root="$(kv_repo_root)"
  export PYTHONPATH="${root}/baseline/crosskv/src${PYTHONPATH:+:${PYTHONPATH}}"
  # Existing baseline package and exact dependencies remain unchanged.
  "${KV_PYTHON}" "${root}/baseline/crosskv/jobs/figure2/check_env.py" \
     --root "${root}/baseline/crosskv" \
     --env "$(cd -- "$(dirname -- "${KV_PYTHON}")/.." && pwd -P)"
  "${KV_PYTHON}" - <<'PY'
import torch
import transformers
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA unavailable; CPU fallback is forbidden')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('ERROR: BF16 required by the fixed Figure 2 config')
if torch.cuda.device_count() != 1:
    raise SystemExit(f'ERROR: expected exactly one visible GPU; got {torch.cuda.device_count()}')
if transformers.__version__ != '4.57.3':
    raise SystemExit('ERROR: Figure 2 requires transformers==4.57.3')
print('RUN GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__)
PY
}

kv_run_figure2() {
  # Arguments: MODE DATASET WORK_DIR RESULT_DIR
  local mode="$1" dataset="$2" work_dir="$3" result_dir="$4" repo_root baseline config
  kv_require_mode "${mode}"
  repo_root="$(kv_repo_root)"
  baseline="${repo_root}/baseline/crosskv"
  config="${baseline}/configs/figure2_qwen3_1p7b_4b.json"
  [[ -f "${dataset}" ]] || { echo "ERROR: dataset missing inside execution environment: ${dataset}" >&2; return 2; }
  [[ -f "${config}" ]] || { echo "ERROR: missing config ${config}" >&2; return 2; }
  [[ -d "${work_dir}" ]] || { echo "ERROR: missing work directory ${work_dir}" >&2; return 2; }
  cd "${baseline}"
  "${KV_PYTHON}" "${repo_root}/run/common/preflight.py" \
    --config "${config}" --dataset "${dataset}" --scratch "${work_dir}" --mode "${mode}" \
    --output "${work_dir}/preflight.json"
  kv_setup_python
  "${KV_PYTHON}" -m pytest -q tests/test_figure2.py tests/test_figure2_tiny_qwen.py
  local smoke_args=()
  if [[ "${mode}" == smoke ]]; then smoke_args+=(--smoke); fi
  "${KV_PYTHON}" -m xmodel_kv.figure2.prepare --config "${config}" \
    --dataset "${dataset}" --run-dir "${work_dir}" "${smoke_args[@]}"
  "${KV_PYTHON}" -m xmodel_kv.figure2.extract --run-dir "${work_dir}" --role source
  "${KV_PYTHON}" -m xmodel_kv.figure2.extract --run-dir "${work_dir}" --role target
  "${KV_PYTHON}" -m xmodel_kv.figure2.probe --run-dir "${work_dir}" --device cuda:0
  "${KV_PYTHON}" -m xmodel_kv.figure2.plot --run-dir "${work_dir}"
  echo "Figure 2 complete: ${work_dir}/results"
}

kv_export_small_results() {
  # Export only small metrics/plots/manifests. Never copy raw activation arrays to NAS.
  local work_dir="$1" result_dir="$2" child
  mkdir -p -- "${result_dir}"
  for child in results provenance; do
    if [[ -d "${work_dir}/${child}" ]]; then
      cp -a -- "${work_dir}/${child}" "${result_dir}/"
    fi
  done
  for child in preflight.json prepare.json; do
    if [[ -f "${work_dir}/${child}" ]]; then
      cp -p -- "${work_dir}/${child}" "${result_dir}/"
    fi
  done
  for child in source target; do
    if [[ -f "${work_dir}/${child}/metadata.json" ]]; then
      mkdir -p -- "${result_dir}/${child}"
      cp -p -- "${work_dir}/${child}/metadata.json" "${result_dir}/${child}/"
    fi
  done
}

kv_record_provenance() {
  local output="$1" platform="$2" mode="$3" repo
  repo="$(kv_repo_root)"
  KV_MANIFEST_PLATFORM="${platform}" KV_MANIFEST_MODE="${mode}" KV_MANIFEST_REPO="${repo}" \
    "${KV_PYTHON}" "${repo}/run/common/manifest.py" --output "${output}"
}
