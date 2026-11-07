#!/usr/bin/env bash
# Execute on the lab-server host. Requires an already running Docker container.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${ROOT}/run/common/runtime.sh"
usage() {
  cat <<'USAGE'
Usage: bash run/lab/figure2.sh --dataset /path/INSIDE/container/file --mode smoke|full [options]
Options:
  --container NAME     Already-running Docker container (default: uav)
  --container-repo DIR Repository location INSIDE container (default: /workspace/kv_cache)
  --gpu 0|1            GPU index as visible INSIDE container (default: 0)
  --tag NAME           Unique run label
  --scratch-root DIR   Writable local scratch INSIDE container (default: /tmp/kv_figure2)
  --result-dir DIR     Persistent output INSIDE container (default: CONTAINER_REPO/results/...)
  --python PATH        Existing Python INSIDE container (default: /usr/local/bin/python)
  --dry-run           Print planned Docker launch without starting it
  --detach            Run in background with docker exec -d (required for long runs)
WARN: container GPU numbering may differ from physical GPU numbering.
USAGE
}
mode=""; dataset=""; tag=""; container="uav"; container_repo="/workspace/kv_cache"
gpu=0; scratch_root="/tmp/kv_figure2"; result_dir=""; python="/usr/local/bin/python"; dry_run=0; detach=0
while (($#)); do
  case "$1" in
    --mode|--dataset|--tag|--container|--container-repo|--gpu|--scratch-root|--result-dir|--python)
      (($# >= 2)) || { usage >&2; exit 2; }
      name="$1"; value="$2"; shift 2
      case "$name" in
        --mode) mode="$value";; --dataset) dataset="$value";; --tag) tag="$value";;
        --container) container="$value";; --container-repo) container_repo="$value";;
        --gpu) gpu="$value";; --scratch-root) scratch_root="$value";;
        --result-dir) result_dir="$value";; --python) python="$value";;
      esac;;
    --dry-run) dry_run=1; shift;;
    --detach) detach=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done
kv_require_mode "$mode"
[[ "$gpu" == 0 || "$gpu" == 1 ]] || { echo "ERROR: GPU must be 0 or 1" >&2; exit 2; }
[[ "$dataset" == /* && "$container_repo" == /* && "$scratch_root" == /* ]] || {
  echo "ERROR: container paths must be absolute" >&2; exit 2;
}
if [[ -z "$tag" ]]; then tag="figure2_${mode}_$(date -u +%Y%m%dT%H%M%SZ)"; fi
kv_require_run_tag "$tag"
[[ -n "$result_dir" ]] || result_dir="${container_repo}/results/crosskv/figure2/${tag}"
[[ "$result_dir" == /* ]] || { echo "ERROR: result-dir must be absolute" >&2; exit 2; }
if [[ "${dry_run}" == 1 ]]; then
  printf "DRY RUN: container=%s repo=%s logical_gpu=%s mode=%s detached=%s\n" "${container}" "${container_repo}" "${gpu}" "${mode}" "${detach}"
  printf "dataset=%s\nresult=%s\nscratch_root=%s\npython=%s\n" "${dataset}" "${result_dir}" "${scratch_root}" "${python}"
  exit 0
fi
kv_require_cmd docker
[[ "$(docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null)" == true ]] || {
  echo "ERROR: Docker container not running: ${container}" >&2; exit 2;
}
# Map the container GPU index to a physical NVML UUID and enforce host GPU 0/1.
kv_require_cmd nvidia-smi
allowed="$(nvidia-smi -i 0,1 --query-gpu=uuid --format=csv,noheader 2>/dev/null)" || {
  echo "ERROR: cannot inspect authorized host GPUs 0 and 1" >&2; exit 2;
}
selected="$(docker exec "$container" nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader 2>/dev/null)" || {
  echo "ERROR: cannot resolve selected container GPU UUID" >&2; exit 2;
}
[[ -n "$selected" ]] && grep -Fxq -- "$selected" <<< "$allowed" || {
  echo "ERROR: selected container GPU is not authorized host GPU 0 or 1" >&2; exit 2;
}
printf "Authorized host GPU UUID: %s (container index %s)\n" "$selected" "$gpu"
# Protect new output directory names without touching existing files.
docker exec --user "$(id -u):$(id -g)" -w "${container_repo}" "${container}" \
  bash -c 'test ! -e "$1"' -- "${result_dir}" || {
    echo "ERROR: result directory already exists inside container: ${result_dir}" >&2; exit 2;
  }
# Docker exec -d is necessary for SSH-independent long experiments.
options=()
if [[ "${detach}" == 1 ]]; then options+=(-d); fi
docker exec "${options[@]}" --user "$(id -u):$(id -g)" \
  -e "CUDA_VISIBLE_DEVICES=${gpu}" -e "KV_PYTHON=${python}" \
  -e "KV_DOCKER_CONTAINER=${container}" -e "KV_FIG2_DETACHED=${detach}" \
  -w "${container_repo}" "${container}" bash run/lab/figure2_inner.sh \
  "$mode" "$dataset" "$tag" "$scratch_root" "$result_dir"
if [[ "${detach}" == 1 ]]; then
  printf 'Detached. Check status: %s/status.txt\nLog: %s/docker.log\n' "${result_dir}" "${result_dir}"
fi
