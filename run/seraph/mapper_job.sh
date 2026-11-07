#!/usr/bin/env bash
# Run exclusively on a compute node allocated by Slurm.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
: "${SLURM_JOB_ID:?Use sbatch through run/seraph/mapper.sh}"
: "${KV_MAPPER_MODE:?}" "${KV_MAPPER_TAG:?}" "${KV_MAPPER_RESULT:?}" "${KV_MAPPER_DIR:?}" "${KV_MAPPER_PYTHON:?}"
export PYTHONPATH="$ROOT:$ROOT/baseline/crosskv/src:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
scratch="${KV_MAPPER_SCRATCH}/${KV_MAPPER_TAG}_job${SLURM_JOB_ID}"
kinds="${KV_MAPPER_KINDS//+/\,}"
[[ ! -e "$scratch" ]] || { echo 'ERROR: scratch already exists' >&2; exit 2; }
mkdir -p "$scratch"
[[ -x "$KV_MAPPER_PYTHON" ]] || { echo 'ERROR: python missing on compute node' >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" != '' ]] || { echo 'ERROR: Slurm did not assign a GPU' >&2; exit 2; }
"$KV_MAPPER_PYTHON" -c 'import torch, transformers, safetensors; assert torch.cuda.is_available(), "no GPU"; print("CUDA:",torch.cuda.get_device_name(0),"Torch:",torch.__version__,"Transformers:",transformers.__version__)'
nvidia-smi > "$KV_MAPPER_RESULT/nvidia-smi.txt"
git -C "$ROOT" rev-parse HEAD > "$KV_MAPPER_RESULT/commit.txt"
git -C "$ROOT" status --short --branch > "$KV_MAPPER_RESULT/git_status.txt"
set +e
status='FAILED'
on_exit() {
  rc=$?
  trap - EXIT
  set +e
  if [[ "$KV_MAPPER_MODE" == train && -d "$scratch/artifacts" ]]; then
    mkdir -p "$KV_MAPPER_DIR"
    # Existing files are protected; config compatibility was validated by Python.
    rsync -a --ignore-existing "$scratch/artifacts/" "$KV_MAPPER_DIR/" || rc=74
  elif [[ "$KV_MAPPER_MODE" == eval && -f "$scratch/evaluation.json" ]]; then
    cp -n "$scratch/evaluation.json" "$KV_MAPPER_RESULT/evaluation.json" || rc=74
  fi
  printf 'status=%s\nexit_code=%s\njob_id=%s\nscratch=%s\n' \
     "$([[ "$rc" == 0 ]] && echo COMPLETE || echo FAILED)" "$rc" "$SLURM_JOB_ID" "$scratch" \
     > "$KV_MAPPER_RESULT/status.txt"
  exit "$rc"
}
trap on_exit EXIT
set -e
if [[ "$KV_MAPPER_MODE" == train ]]; then
  mkdir -p "$KV_MAPPER_DIR"
  exec 9>"$KV_MAPPER_DIR/.train.lock"
  flock -n 9 || { echo "ERROR: another training job uses the same mapper output directory" >&2; exit 2; }
  # Excludes all unselected source layers and all untargeted layers.
  "$KV_MAPPER_PYTHON" -m proposed.exp.mapper.stage \
     --source "$KV_MAPPER_SOURCE" --target "$KV_MAPPER_TARGET" \
     --selection "$KV_MAPPER_SELECTION" --output "$scratch/calibration" \
     --layer-start "$KV_MAPPER_LAYER_START" --layer-end "$KV_MAPPER_LAYER_END" \
     --kinds "$kinds"
  mkdir -p "$scratch/artifacts"
  # Only copy existing config + currently requested layer files, not unrelated layers.
  if [[ -f "$KV_MAPPER_DIR/config.json" ]]; then
    cp -n "$KV_MAPPER_DIR/config.json" "$scratch/artifacts/config.json"
  fi
  for ((layer=KV_MAPPER_LAYER_START; layer<KV_MAPPER_LAYER_END; layer++)); do
    printf -v ldir 'layer_%03d' "$layer"
    if [[ -d "$KV_MAPPER_DIR/layers/$ldir" ]]; then
      mkdir -p "$scratch/artifacts/layers/$ldir"
      rsync -a "$KV_MAPPER_DIR/layers/$ldir/" "$scratch/artifacts/layers/$ldir/"
    fi
  done
  args=(train --source-dir "$scratch/calibration/source" \
       --target-dir "$scratch/calibration/target" \
       --selection "$scratch/calibration/selection.json" \
       --output "$scratch/artifacts" --layer-start "$KV_MAPPER_LAYER_START" \
       --layer-end "$KV_MAPPER_LAYER_END" --head-start "$KV_MAPPER_HEAD_START" \
       --epochs "$KV_MAPPER_EPOCHS" --batch-size "$KV_MAPPER_BATCH" \
       --hidden-dim "$KV_MAPPER_HIDDEN" --lr "$KV_MAPPER_LR" \
       --seed "$KV_MAPPER_SEED" --kinds "$kinds" \
       --train-dtype "$KV_MAPPER_TRAIN_DTYPE" --storage-dtype "$KV_MAPPER_STORAGE_DTYPE")
  [[ -z "$KV_MAPPER_HEAD_END" ]] || args+=(--head-end "$KV_MAPPER_HEAD_END")
  [[ -z "$KV_MAPPER_MAX_OBS" ]] || args+=(--max-observations "$KV_MAPPER_MAX_OBS")
  "$KV_MAPPER_PYTHON" -m proposed.exp.mapper.cli "${args[@]}"
elif [[ "$KV_MAPPER_MODE" == eval ]]; then
  mkdir -p "$scratch/mapper"
  rsync -a "$KV_MAPPER_DIR/" "$scratch/mapper/"
  cp -p "$KV_MAPPER_INPUT" "$scratch/eval.jsonl"
  "$KV_MAPPER_PYTHON" -m proposed.exp.mapper.cli eval \
      --mapper-dir "$scratch/mapper" --input-jsonl "$scratch/eval.jsonl" \
      --output-json "$scratch/evaluation.json" --prefix-length "$KV_MAPPER_PREFIX" \
      --max-examples "$KV_MAPPER_EXAMPLES" --token-chunk "$KV_MAPPER_TOKEN_CHUNK"
else
  echo 'ERROR: unknown mode' >&2; exit 2
fi
