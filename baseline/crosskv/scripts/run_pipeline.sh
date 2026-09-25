#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${XKV_SOURCE_MODEL:=Qwen/Qwen3-8B}"
: "${XKV_TARGET_MODEL:=Qwen/Qwen3-32B}"
: "${XKV_CALIBRATION_DATA:?Set XKV_CALIBRATION_DATA in .env or the environment}"
: "${XKV_TEXT_FIELD:=text}"
: "${XKV_WORK_DIR:=work/qwen3_8b_to_32b}"
: "${XKV_NUM_SEQUENCES:=500}"
: "${XKV_SEQUENCE_LENGTH:=1024}"
: "${XKV_STRIDE:=4}"
: "${XKV_TOP_K:=12}"
: "${XKV_RIDGE:=0.01}"
: "${XKV_WEIGHT_DTYPE:=float32}"
: "${XKV_DEVICE:=cuda:0}"
: "${XKV_DEVICE_MAP:=auto}"
: "${XKV_ATTN_IMPLEMENTATION:=sdpa}"

if command -v uv >/dev/null 2>&1; then
  run=(uv run)
elif command -v xkv-prepare >/dev/null 2>&1; then
  run=()
else
  echo "Install uv or activate an environment containing xmodel-kv." >&2
  exit 1
fi

"${run[@]}" xkv-prepare \
  --dataset "${XKV_CALIBRATION_DATA}" \
  --tokenizer "${XKV_SOURCE_MODEL}" \
  --output "${XKV_WORK_DIR}/tokens.npy" \
  --text-field "${XKV_TEXT_FIELD}" \
  --num-sequences "${XKV_NUM_SEQUENCES}" \
  --sequence-length "${XKV_SEQUENCE_LENGTH}"

"${run[@]}" xkv-extract \
  --model "${XKV_SOURCE_MODEL}" \
  --tokens "${XKV_WORK_DIR}/tokens.npy" \
  --output "${XKV_WORK_DIR}/source" \
  --role source \
  --stride "${XKV_STRIDE}" \
  --device-map "${XKV_DEVICE_MAP}" \
  --attn-implementation "${XKV_ATTN_IMPLEMENTATION}"

"${run[@]}" xkv-extract \
  --model "${XKV_TARGET_MODEL}" \
  --tokens "${XKV_WORK_DIR}/tokens.npy" \
  --output "${XKV_WORK_DIR}/target" \
  --role target \
  --stride "${XKV_STRIDE}" \
  --device-map "${XKV_DEVICE_MAP}" \
  --attn-implementation "${XKV_ATTN_IMPLEMENTATION}"

"${run[@]}" xkv-select \
  --source "${XKV_WORK_DIR}/source" \
  --target "${XKV_WORK_DIR}/target" \
  --output "${XKV_WORK_DIR}/selection_k${XKV_TOP_K}.json" \
  --top-k "${XKV_TOP_K}" \
  --ridge 0 \
  --device "${XKV_DEVICE}"

"${run[@]}" xkv-fit \
  --source "${XKV_WORK_DIR}/source" \
  --target "${XKV_WORK_DIR}/target" \
  --selection "${XKV_WORK_DIR}/selection_k${XKV_TOP_K}.json" \
  --output "${XKV_WORK_DIR}/mapper_k${XKV_TOP_K}" \
  --ridge "${XKV_RIDGE}" \
  --device "${XKV_DEVICE}" \
  --weight-dtype "${XKV_WEIGHT_DTYPE}"

echo "Mapper written to ${XKV_WORK_DIR}/mapper_k${XKV_TOP_K}"
