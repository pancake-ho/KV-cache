#!/usr/bin/env bash
# Legacy Seraph-only launcher. Existing LoRA source and configs stay unchanged.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/../../.." && pwd -P)"
PROJECT="$ROOT/baseline/lora_exp"
EXPECTED="/data/surt321/repos/lab/kv_cache"
mode="--dry-run"
if (( $# > 0 )); then mode="$1"; fi
case "$mode" in
  --dry-run|--submit) ;;
  -h|--help) echo "Usage: bash run/seraph/baseline/lora_qkv.sh --dry-run|--submit"; exit 0 ;;
  *) echo "ERROR: supported options --dry-run or --submit" >&2; exit 2 ;;
esac
(( $# <= 1 )) || { echo "ERROR: unexpected arguments" >&2; exit 2; }
[[ "$ROOT" == "$EXPECTED" ]] || {
  echo "ERROR: existing LoRA configs use Seraph paths at $EXPECTED; got $ROOT" >&2
  exit 2
}
[[ -f "$PROJECT/slurm/characterize_qkv.sh" ]] || {
  echo "ERROR: missing legacy LoRA script" >&2; exit 2
}
echo "Legacy QKV job: 1 GPU, 8 CPUs, 18G RAM, partition batch_eebme_ugrad"
echo "No config, model adapter, dataset or existing result is changed."
if [[ "$mode" == "--dry-run" ]]; then
  echo "DRY RUN: cd '$PROJECT' && sbatch --chdir='$PROJECT' slurm/characterize_qkv.sh"
  exit 0
fi
command -v sbatch >/dev/null || { echo "ERROR: sbatch not found" >&2; exit 2; }
mkdir -p -- "$PROJECT/runs/qkv"
cd "$PROJECT"
sbatch --chdir="$PROJECT" slurm/characterize_qkv.sh
