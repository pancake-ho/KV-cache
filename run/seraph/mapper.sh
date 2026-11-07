#!/usr/bin/env bash
# Submit one GPU job for training or evaluation of the neural KV mapper.
# Never edits baseline/, never cancels jobs, and never erases scratch/results.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
usage() {
  cat <<'TXT'
Usage:
  bash run/seraph/mapper.sh train --source-dir /data/.../source --target-dir /data/.../target \
      --selection /data/.../selection.json --mapper-dir /data/.../mapper \
      --layer-start 0 --layer-end 1 --tag YOUR_UNIQUE_TAG [--head-start 0 --head-end 8] \
      [--max-observations 2048 --epochs 1 --batch-size 256 --kinds k,v] [--dry-run]
  bash run/seraph/mapper.sh eval --mapper-dir /data/.../mapper \
      --input-jsonl /data/.../eval.jsonl --tag YOUR_UNIQUE_TAG \
      [--prefix-length 128 --max-examples 4 --token-chunk 128] [--dry-run]
Optional: --cpus 8 --mem 29G --time 1-00:00:00 --partition batch_eebme_ugrad
          --result-dir /data/.../result-dir --scratch-root /local_datasets/$USER/kv_mapper
          --python /data/$USER/anaconda3/envs/lab/bin/python
The per-head fit uses a single assigned GPU; run stages large arrays onto local disk.
TXT
}
[[ $# -ge 1 ]] || { usage; exit 2; }
mode="$1"; shift
[[ "$mode" == train || "$mode" == eval ]] || { usage; exit 2; }
source_dir='' target_dir='' selection='' mapper_dir='' input_jsonl='' result_dir=''
tag='' layer_start='' layer_end='' head_start=0 head_end='' max_obs='' epochs=20 batch_size=4096
kinds='k,v' hidden_dim=1024 lr=0.001 seed=2026 train_dtype='bfloat16' storage_dtype='float32'
prefix_length=128 max_examples=4 token_chunk=128
partition='batch_eebme_ugrad'; cpus=8; mem=29G; walltime='1-00:00:00'
scratch_root="/local_datasets/${USER}/kv_mapper"; python="/data/${USER}/anaconda3/envs/lab/bin/python"
dry_run=0
while (( $# )); do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --source-dir|--target-dir|--selection|--mapper-dir|--input-jsonl|--result-dir|--tag|--layer-start|--layer-end|--head-start|--head-end|--max-observations|--epochs|--batch-size|--hidden-dim|--lr|--seed|--train-dtype|--storage-dtype|--kinds|--prefix-length|--max-examples|--token-chunk|--partition|--cpus|--mem|--time|--scratch-root|--python)
      (( $# >= 2 )) || { usage >&2; exit 2; }
      case "$1" in
        --source-dir) source_dir="$2";; --target-dir) target_dir="$2";;
        --selection) selection="$2";; --mapper-dir) mapper_dir="$2";;
        --input-jsonl) input_jsonl="$2";; --result-dir) result_dir="$2";;
        --tag) tag="$2";; --layer-start) layer_start="$2";;
        --layer-end) layer_end="$2";; --head-start) head_start="$2";;
        --head-end) head_end="$2";; --max-observations) max_obs="$2";;
        --epochs) epochs="$2";; --batch-size) batch_size="$2";;
        --hidden-dim) hidden_dim="$2";; --lr) lr="$2";;
        --seed) seed="$2";; --train-dtype) train_dtype="$2";;
        --storage-dtype) storage_dtype="$2";; --kinds) kinds="$2";;
        --prefix-length) prefix_length="$2";; --max-examples) max_examples="$2";;
        --token-chunk) token_chunk="$2";; --partition) partition="$2";;
        --cpus) cpus="$2";; --mem) mem="$2";;
        --time) walltime="$2";; --scratch-root) scratch_root="$2";;
        --python) python="$2";;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$ ]] || { echo 'ERROR: supply a safe unique --tag' >&2; exit 2; }
[[ "$scratch_root" == /local_datasets/* ]] || { echo 'ERROR: scratch must be compute-node-local' >&2; exit 2; }
[[ "$cpus" =~ ^([1-9]|1[0-6])$ && "$mem" == 29G ]] || { echo 'ERROR: undergrad limits are CPU <=16, RAM=29G' >&2; exit 2; }
[[ "$mapper_dir" == /data/* && "$mapper_dir" != "$ROOT/baseline/"* ]] || { echo 'ERROR: --mapper-dir must be an absolute writable path under /data (not baseline)' >&2; exit 2; }
if [[ "$mode" == train ]]; then
  [[ "$source_dir" == /data/* && "$target_dir" == /data/* && "$selection" == /data/* ]] || { echo 'ERROR: source, target, selection must use /data absolute paths' >&2; exit 2; }
  [[ "$layer_start" =~ ^[0-9]+$ && "$layer_end" =~ ^[1-9][0-9]*$ && "$layer_end" -gt "$layer_start" ]] || { echo 'ERROR: valid --layer-start/end required' >&2; exit 2; }
  [[ "$head_start" =~ ^[0-9]+$ && ( -z "$head_end" || "$head_end" =~ ^[1-9][0-9]*$ ) ]] || { echo 'ERROR: head range invalid' >&2; exit 2; }
  for path in "$source_dir/metadata.json" "$target_dir/metadata.json" "$selection"; do
    [[ -f "$path" ]] || { echo "ERROR: missing $path" >&2; exit 2; }
  done
else
  [[ "$input_jsonl" == /data/* && -f "$input_jsonl" ]] || { echo 'ERROR: --input-jsonl must exist under /data' >&2; exit 2; }
  [[ -f "$mapper_dir/config.json" ]] || { echo 'ERROR: --mapper-dir missing config.json' >&2; exit 2; }
fi
[[ -n "$result_dir" ]] || result_dir="$ROOT/results/mapper/$tag"
[[ "$result_dir" == /data/* && "$result_dir" != "$mapper_dir" ]] || { echo 'ERROR: result directory must be unique under /data' >&2; exit 2; }
[[ ! -e "$result_dir" ]] || { echo "ERROR: result directory exists: $result_dir" >&2; exit 2; }
if [[ "$dry_run" == 1 ]]; then
  printf 'DRY RUN: mode=%s tag=%s partition=%s gpu=1 cpus=%s mem=%s time=%s\n' "$mode" "$tag" "$partition" "$cpus" "$mem" "$walltime"
  printf 'source=%s target=%s selection=%s mapper=%s inputs=%s\n' "$source_dir" "$target_dir" "$selection" "$mapper_dir" "$input_jsonl"
  printf 'result=%s scratch=%s python=%s\n' "$result_dir" "$scratch_root" "$python"
  exit 0
fi
command -v sbatch >/dev/null && command -v sinfo >/dev/null || { echo 'ERROR: Slurm tools missing' >&2; exit 2; }
[[ -x "$python" ]] || { echo "ERROR: python not executable: $python" >&2; exit 2; }
[[ -n "$(sinfo -h -p "$partition" -o '%P')" ]] || { echo 'ERROR: unavailable partition' >&2; exit 2; }
mkdir -p "$result_dir/slurm"
# Environment variables avoid command injection and preserve user-provided paths.
job="$(sbatch --parsable --job-name="kv-mlp-${mode}" --partition="$partition" \
   --gres=gpu:1 --cpus-per-task="$cpus" --mem="$mem" --time="$walltime" \
   --chdir="$ROOT" --output="$result_dir/slurm/%x_%j.out" \
   --error="$result_dir/slurm/%x_%j.err" \
   --export="ALL,KV_MAPPER_MODE=$mode,KV_MAPPER_TAG=$tag,KV_MAPPER_SOURCE=$source_dir,KV_MAPPER_TARGET=$target_dir,KV_MAPPER_SELECTION=$selection,KV_MAPPER_DIR=$mapper_dir,KV_MAPPER_INPUT=$input_jsonl,KV_MAPPER_RESULT=$result_dir,KV_MAPPER_SCRATCH=$scratch_root,KV_MAPPER_PYTHON=$python,KV_MAPPER_LAYER_START=$layer_start,KV_MAPPER_LAYER_END=$layer_end,KV_MAPPER_HEAD_START=$head_start,KV_MAPPER_HEAD_END=$head_end,KV_MAPPER_MAX_OBS=$max_obs,KV_MAPPER_EPOCHS=$epochs,KV_MAPPER_BATCH=$batch_size,KV_MAPPER_KINDS=${kinds//,/+},KV_MAPPER_HIDDEN=$hidden_dim,KV_MAPPER_LR=$lr,KV_MAPPER_SEED=$seed,KV_MAPPER_TRAIN_DTYPE=$train_dtype,KV_MAPPER_STORAGE_DTYPE=$storage_dtype,KV_MAPPER_PREFIX=$prefix_length,KV_MAPPER_EXAMPLES=$max_examples,KV_MAPPER_TOKEN_CHUNK=$token_chunk" \
   "$ROOT/run/seraph/mapper_job.sh")"
job="${job%%;*}"
printf 'Submitted job %s\nResults: %s\nMonitor: squeue -j %s\n' "$job" "$result_dir" "$job"
