#!/usr/bin/env bash
# Stable baseline entry point. Do not duplicate algorithm logic here.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/../../.." && pwd -P)"
exec bash "$ROOT/run/seraph/figure2.sh" "$@"
