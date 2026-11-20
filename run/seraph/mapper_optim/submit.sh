#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
method="$1"; shift
usage() {
  printf '%s\n' "Usage: bash run/seraph/$method/run.sh [--dry-run] FIT_OR_EVAL TAG [Python options]" \
    "Example help: bash run/seraph/$method/run.sh fit --help" \
    "Options via environment: KV_OPT_PYTHON, KV_OPT_RESULTS, KV_OPT_SCRATCH, KV_OPT_PARTITION," \
    "                         KV_OPT_CPUS, KV_OPT_MEM, KV_OPT_TIME"
}
seraph_user="${USER:-$(id -un)}"
python="${KV_OPT_PYTHON:-/data/$seraph_user/anaconda3/envs/lab/bin/python}"
dry=0
if [[ "${1:-}" == --dry-run ]]; then dry=1; shift; fi
stage="${1:-}"
case "$stage" in
  -h|--help|'') usage; exit 0 ;;
  fit|eval) shift ;;
  *) usage >&2; exit 2 ;;
esac
export PYTHONPATH="$root:$root/baseline/crosskv/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  exec "$python" -m "proposed.src.exp.$method.cli" "$stage" --help
fi
tag="${1:-}"
[[ "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]] || { usage >&2; exit 2; }
shift
for arg in "$@"; do
  case "$arg" in
    --output*|-h|--help|--) echo "Option reserved by launcher: $arg" >&2; exit 2 ;;
  esac
done
partition="${KV_OPT_PARTITION:-batch_eebme_ugrad}"
cpus="${KV_OPT_CPUS:-8}"
memory="${KV_OPT_MEM:-29G}"
walltime="${KV_OPT_TIME:-04:00:00}"
scratch="${KV_OPT_SCRATCH:-/local_datasets/$seraph_user/kv_mapper_optim}"
results="${KV_OPT_RESULTS:-/data/$seraph_user/kv_mapper_optim_runs}"
[[ "$cpus" =~ ^([1-9]|1[0-6])$ && "$memory" =~ ^([1-9]|[12][0-9])G$ ]] || {
  echo "Expected 1–16 CPUs and 1–29G system RAM" >&2; exit 2;
}
[[ "$results" == /* && "$scratch" == /* && "$python" == /* ]] || {
  echo "Results, scratch and Python paths must be absolute" >&2; exit 2;
}
result="$results/$method/$tag"
cmd=(sbatch --parsable "--job-name=kv-$method-$stage" "--partition=$partition"
     --gres=gpu:1 "--cpus-per-task=$cpus" "--mem=$memory" "--time=$walltime"
     "--chdir=$root" "--output=$result/slurm-%j.out"
     "$root/run/seraph/mapper_optim/job.sh" "$root" "$method" "$python" "$result"
     "$scratch" "$stage" "$@")
if (( dry )); then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi
[[ -x "$python" ]] || { echo "Python is unavailable: $python" >&2; exit 2; }
command -v sinfo >/dev/null && command -v sbatch >/dev/null || {
  echo "Run this launcher from a Seraph Slurm login node" >&2; exit 2;
}
[[ -n "$(sinfo -h -p "$partition" -o '%P')" ]] || {
  echo "Partition unavailable: $partition" >&2; exit 2;
}
mkdir -p -- "$results/$method"
mkdir -- "$result" || { echo "Use a new result tag: $result" >&2; exit 2; }
git -C "$root" rev-parse HEAD > "$result/commit.txt"
git -C "$root" status --short --branch > "$result/git_status.txt"
printf '%q ' "${cmd[@]}" > "$result/command.txt"
printf '\n' >> "$result/command.txt"
if job="$("${cmd[@]}")"; then
  printf '%s\n' "${job%%;*}" > "$result/job_id.txt"
  printf 'Job %s\nResults %s\n' "${job%%;*}" "$result"
else
  rc=$?
  printf 'status=SUBMIT_FAILED\nexit_code=%s\n' "$rc" > "$result/status.txt"
  exit "$rc"
fi
