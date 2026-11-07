#!/usr/bin/env bash
# Run on moana-master. Submits one Slurm job for all Figure-2 stages.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${ROOT}/run/common/runtime.sh"

usage() {
  cat <<'USAGE'
Usage: bash run/seraph/figure2.sh --dataset /data/.../fineweb.parquet --mode smoke|full [options]
Options:
  --tag TAG          Unique experiment label (default: dated figure2_<mode>_...)
  --partition NAME   Slurm partition (default: batch_eebme_ugrad)
  --cpus N           CPU allocation (default: 16)
  --mem SIZE         RAM allocation (default: 29G)
  --time TIME        Slurm walltime (default: 1-00:00:00)
  --result-dir PATH  Persistent small-result directory on /data (default: REPO/results/...)
  --scratch-root PATH Compute-node-local scratch (default: /local_datasets/$USER/kv_figure2)
  --python PATH      Compute-node interpreter (default: /data/$USER/anaconda3/envs/lab/bin/python)
  --dry-run          Print planned Slurm launch without submission or output creation
Note: dataset must be accessible from compute node; one file is staged to scratch.
USAGE
}
mode=""; dataset=""; tag=""; partition="batch_eebme_ugrad"; cpus=16; mem=29G
walltime="1-00:00:00"; result_dir=""; scratch_root="/local_datasets/${USER}/kv_figure2"
python="/data/${USER}/anaconda3/envs/lab/bin/python"; dry_run=0
while (($#)); do
  case "$1" in
    --mode|--dataset|--tag|--partition|--cpus|--mem|--time|--result-dir|--scratch-root|--python)
      (($# >= 2)) || { usage >&2; exit 2; }
      name="$1"; value="$2"; shift 2
      case "${name}" in
        --mode) mode="$value";; --dataset) dataset="$value";; --tag) tag="$value";;
        --partition) partition="$value";; --cpus) cpus="$value";; --mem) mem="$value";;
        --time) walltime="$value";; --result-dir) result_dir="$value";;
        --scratch-root) scratch_root="$value";; --python) python="$value";;
      esac;;
    --dry-run) dry_run=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done
kv_require_mode "${mode}"
[[ -f "${dataset}" ]] || { echo "ERROR: dataset not found: ${dataset}" >&2; exit 2; }
[[ "${dataset}" == /data/* ]] || { echo "ERROR: Seraph dataset should be on shared /data so allocated jobs can stage it" >&2; exit 2; }
[[ "${scratch_root}" == /local_datasets/* ]] || { echo "ERROR: scratch-root must be on compute-node /local_datasets" >&2; exit 2; }
[[ "${cpus}" =~ ^([1-9]|1[0-6])$ ]] || { echo "ERROR: undergrad CPU allocation must be 1..16" >&2; exit 2; }
[[ "${mem}" == 29G ]] || { echo "ERROR: fixed undergrad memory limit for this launcher is 29G" >&2; exit 2; }
if [[ -z "${tag}" ]]; then tag="figure2_${mode}_$(date -u +%Y%m%dT%H%M%SZ)"; fi
kv_require_run_tag "${tag}"
dataset="$(realpath -- "${dataset}")"
[[ -n "${result_dir}" ]] || result_dir="${ROOT}/results/crosskv/figure2/${tag}"
[[ "${result_dir}" == /* ]] || { echo "ERROR: result-dir must be absolute" >&2; exit 2; }
[[ "${result_dir}" == /data/* ]] || { echo "ERROR: Seraph result-dir must be under /data" >&2; exit 2; }
[[ ! -e "${result_dir}" ]] || { echo "ERROR: result directory already exists: ${result_dir}" >&2; exit 2; }
if [[ "${dry_run}" == 1 ]]; then
  printf "DRY RUN: mode=%s partition=%s gpu=1 cpus=%s mem=%s time=%s\n" "${mode}" "${partition}" "${cpus}" "${mem}" "${walltime}"
  printf "dataset=%s\nresult=%s\nscratch_root=%s\npython=%s\n" "${dataset}" "${result_dir}" "${scratch_root}" "${python}"
  exit 0
fi
kv_require_cmd sbatch
kv_require_cmd sinfo
[[ -n "$(sinfo -h -p "${partition}" -o '%P')" ]] || { echo "ERROR: unknown Slurm partition: ${partition}" >&2; exit 2; }
mkdir -p -- "${result_dir}/slurm"
# Submission exports only the parameters; model load and numeric work happen on compute nodes.
job="$(sbatch --parsable --job-name="kv-fig2-${mode}" \
  --partition="${partition}" --gres=gpu:1 --cpus-per-task="${cpus}" \
  --mem="${mem}" --time="${walltime}" --chdir="${ROOT}" \
  --output="${result_dir}/slurm/%x_%j.out" --error="${result_dir}/slurm/%x_%j.err" \
  --export="ALL,KV_FIG2_MODE=${mode},KV_FIG2_DATASET=${dataset},KV_FIG2_TAG=${tag},KV_FIG2_RESULT_DIR=${result_dir},KV_FIG2_SCRATCH_ROOT=${scratch_root},KV_PYTHON=${python}" \
  "${ROOT}/run/seraph/figure2_job.sh")"
job="${job%%;*}"
printf 'Submitted Slurm job: %s\nResult directory: %s\nMonitor: squeue -j %s\n' "${job}" "${result_dir}" "${job}"
