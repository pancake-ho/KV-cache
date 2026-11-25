#!/usr/bin/env bash
set -euo pipefail
root="$1" method="$2" python="$3" result="$4" scratch="$5" stage="$6"
shift 6
: "${SLURM_JOB_ID:?Launch through sbatch}"
work=''
publish() {
  rc=$?
  trap - EXIT
  if (( rc == 0 )); then
    if [[ -d "$work/output" ]] && cp -a -- "$work/output" "$result/.output-$SLURM_JOB_ID" \
        && mv -- "$result/.output-$SLURM_JOB_ID" "$result/output"; then
      :
    else
      rc=74
    fi
  fi
  if [[ -n "$work" && -d "$work" && -n "${SLURM_JOB_ID:-}" ]]; then
    rm -rf -- "$work"
  fi
  printf 'status=%s\nexit_code=%s\njob_id=%s\n' \
    "$([[ "$rc" == 0 ]] && echo COMPLETE || echo FAILED)" "$rc" "$SLURM_JOB_ID" > "$result/status.txt"
  exit "$rc"
}
trap publish EXIT
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose a GPU}"
export PYTHONPATH="$root:$root/baseline/crosskv/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd -- "$root"
nvidia-smi > "$result/nvidia-smi.txt"
mkdir -p -- "$scratch"
work="$(mktemp -d "$scratch/$method-${SLURM_JOB_ID}-XXXXXXXX")"
"$python" -m "proposed.src.exp.$method.cli" "$stage" "$@" --output "$work/output"
