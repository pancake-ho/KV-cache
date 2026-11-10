#!/usr/bin/env bash
#SBATCH --job-name=qwen3_serving
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodelist=moana-y5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=29G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
: "${SLURM_JOB_ID:?submit through Slurm}" "${CUDA_VISIBLE_DEVICES:?allocate one GPU}"
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"

conda_base=$(conda info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate semcache-qwen
set -u
SEMCACHE_PYTHON=$(command -v python)

exec {config_fd}< <(
    "$SEMCACHE_PYTHON" - <<'PY'
import os
import sys
from pathlib import Path

from proposed.validation import config


def path(value):
    return "" if value is None else str((config.REPO_ROOT / Path(value).expanduser()).resolve())


mlp = path(config.MLP_DIR)
ridge = path(config.RIDGE_DIR)
if not mlp and not ridge:
    sys.exit("Set MLP_DIR or RIDGE_DIR in proposed/validation/config.py")
if len(config.LORA_ADAPTERS) not in (0, 2):
    sys.exit("Configure zero or two LoRA adapters")

args = [
    "--prefix-tokens", str(config.PREFIX_TOKENS),
    "--suffix-tokens", str(config.SUFFIX_TOKENS),
    "--new-tokens", str(config.NEW_TOKENS),
]
for name, directory in config.LORA_ADAPTERS.items():
    args.extend(("--adapter", name, path(directory)))
if config.ALLOW_DOWNLOAD:
    args.append("--allow-download")

fields = [path(config.OUTPUT_ROOT), mlp, ridge, *args]
sys.stdout.buffer.write(b"".join(os.fsencode(field) + b"\0" for field in fields))
PY
)
config_pid=$!
mapfile -d '' -t settings <&"$config_fd"
wait "$config_pid"
exec {config_fd}<&-

SEMCACHE_OUT="${settings[0]}/serving_${SLURM_JOB_ID}"
SEMCACHE_MLP="${settings[1]}"
SEMCACHE_RIDGE="${settings[2]}"
args=("${settings[@]:3}" --output "$SEMCACHE_OUT/summary.json")

mkdir -p -- "${settings[0]}"
mkdir -- "$SEMCACHE_OUT"
printf '%s\n' "$SLURM_JOB_ID" > "$SEMCACHE_OUT/job_id.txt"
echo "Results: $SEMCACHE_OUT"

status=FAILED
trap 'printf "status=%s\nexit_code=%s\njob_id=%s\n" "$status" "$?" "$SLURM_JOB_ID" > "$SEMCACHE_OUT/status.txt"' EXIT
export TOKENIZERS_PARALLELISM=false
"$SEMCACHE_PYTHON" -c 'import torch; assert torch.cuda.device_count() == 1, "Exactly one CUDA GPU is required"'

local_root="/local_datasets/$USER/kv_mapper"
mkdir -p "$local_root"
staged=$(mktemp -d "$local_root/semcache_${SLURM_JOB_ID}_XXXXXX")
echo "Staging mappers in $staged"

if [[ -n "${SEMCACHE_MLP:-}" ]]; then
    test -s "$SEMCACHE_MLP/config.json"
    rsync -a -- "$SEMCACHE_MLP/" "$staged/mlp/"
    args+=(--mlp-dir "$staged/mlp")
fi
if [[ -n "${SEMCACHE_RIDGE:-}" ]]; then
    test -s "$SEMCACHE_RIDGE/config.json"
    rsync -a -- "$SEMCACHE_RIDGE/" "$staged/ridge/"
    args+=(--ridge-dir "$staged/ridge")
fi

"$SEMCACHE_PYTHON" -u -m proposed.validation.semcache_smoke "${args[@]}"
status=COMPLETE
