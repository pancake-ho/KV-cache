#!/usr/bin/env bash
# Run inside the compute-node allocation where the namespace/veth pair exists.
set -euo pipefail

mode=${1:?usage: qwen3_ud_es_tc.sh smoke|bench [benchmark options]}
shift
case "$mode" in
  smoke|bench) ;;
  *) echo "Expected smoke or bench" >&2; exit 2 ;;
esac
: "${KV_TC_ES_NS:?Set KV_TC_ES_NS to the existing ES namespace}"
: "${KV_TC_UD_NS:?Set KV_TC_UD_NS to the existing UD namespace}"
: "${KV_TC_ES_IP:?Set KV_TC_ES_IP to the ES veth IPv4 address}"
: "${KV_TC_UD_IP:?Set KV_TC_UD_IP to the UD veth IPv4 address}"

cd "$(dirname -- "$0")/../.."
python_bin=$(command -v python)
run_ns() {
  local ns=$1
  shift
  if [[ -n ${KV_TC_NS_RUNNER:-} ]]; then
    "$KV_TC_NS_RUNNER" "$ns" "$@"
  else
    ip netns exec "$ns" "$@"
  fi
}

mkdir -p results/e2e
result_dir=$(mktemp -d "$PWD/results/e2e/ud_es_${SLURM_JOB_ID:-manual}_XXXXXX")
scratch_dir=$(mktemp -d /tmp/ud-es.XXXXXX)
es_pid=''
cleanup() {
  if [[ -n $es_pid ]]; then
    kill "$es_pid" 2>/dev/null || true
    wait "$es_pid" 2>/dev/null || true
  fi
  rm -f -- "$scratch_dir/es.sock"
  rmdir -- "$scratch_dir"
}
trap cleanup EXIT
echo "Results $result_dir"

snapshot() {
  local name=$1 ns=$2 local_ip=$3 peer_ip=$4 phase=$5
  run_ns "$ns" "$python_bin" -m proposed.communication.metrics.netem \
    "$local_ip" "$peer_ip" "$result_dir/$name" "$phase"
}

snapshot es "$KV_TC_ES_NS" "$KV_TC_ES_IP" "$KV_TC_UD_IP" before
snapshot ud "$KV_TC_UD_NS" "$KV_TC_UD_IP" "$KV_TC_ES_IP" before
common=(
  --es-ip "$KV_TC_ES_IP" --ud-ip "$KV_TC_UD_IP"
  --port "${KV_TC_PORT:-38197}" --rendezvous "$scratch_dir/es.sock"
)
run_ns "$KV_TC_ES_NS" "$python_bin" -m proposed.communication.benchmark es "${common[@]}" &
es_pid=$!
run_ns "$KV_TC_UD_NS" "$python_bin" -u -m proposed.communication.benchmark \
  "$mode" "${common[@]}" "$@" --out "$result_dir/report.jsonl"
wait "$es_pid"
es_pid=''
snapshot es "$KV_TC_ES_NS" "$KV_TC_ES_IP" "$KV_TC_UD_IP" after
snapshot ud "$KV_TC_UD_NS" "$KV_TC_UD_IP" "$KV_TC_ES_IP" after
