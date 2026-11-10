#!/usr/bin/env bash
# Submit with sbatch from the KV-cache repository root.
set -euo pipefail
: "${SLURM_JOB_ID:?submit through Slurm}" "${CUDA_VISIBLE_DEVICES:?allocate one GPU}"
: "${SEMCACHE_OUT:?export the result directory}" "${SEMCACHE_PYTHON:?export the Python interpreter}"
[[ -d "$SEMCACHE_OUT" && ! -e "$SEMCACHE_OUT/summary.json" ]] || {
    echo "Use an existing result directory without a summary.json" >&2
    exit 2
}
if [[ -z "${SEMCACHE_MLP:-}" && -z "${SEMCACHE_RIDGE:-}" ]]; then
    echo "Set SEMCACHE_MLP or SEMCACHE_RIDGE to an existing mapper directory" >&2
    exit 2
fi
if [[ -n "${SEMCACHE_LORA_A:-}" || -n "${SEMCACHE_LORA_B:-}" ]]; then
    : "${SEMCACHE_LORA_A:?both LoRA directories are required}" "${SEMCACHE_LORA_B:?both LoRA directories are required}"
fi

status=FAILED
trap 'printf "status=%s\nexit_code=%s\njob_id=%s\n" "$status" "$?" "$SLURM_JOB_ID" > "$SEMCACHE_OUT/status.txt"' EXIT
cd "$(git rev-parse --show-toplevel)"
export TOKENIZERS_PARALLELISM=false
"$SEMCACHE_PYTHON" -c 'import torch; assert torch.cuda.device_count() == 1, "Exactly one CUDA GPU is required"'

local_root="/local_datasets/$USER/kv_mapper"
mkdir -p "$local_root"
staged=$(mktemp -d "$local_root/semcache_${SLURM_JOB_ID}_XXXXXX")
echo "Staging mappers in $staged"
args=(
    --prefix-tokens "${SEMCACHE_PREFIX_TOKENS:-512}"
    --suffix-tokens "${SEMCACHE_SUFFIX_TOKENS:-24}"
    --new-tokens "${SEMCACHE_NEW_TOKENS:-8}"
    --output "$SEMCACHE_OUT/summary.json"
)
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
if [[ -n "${SEMCACHE_LORA_A:-}" ]]; then
    args+=(--adapter "${SEMCACHE_LORA_A_ID:-lora-a}" "$SEMCACHE_LORA_A"
           --adapter "${SEMCACHE_LORA_B_ID:-lora-b}" "$SEMCACHE_LORA_B")
fi
if [[ "${SEMCACHE_ALLOW_DOWNLOAD:-0}" == 1 ]]; then
    args+=(--allow-download)
fi
"$SEMCACHE_PYTHON" -u -m proposed.validation.semcache_smoke "${args[@]}"
status=COMPLETE
