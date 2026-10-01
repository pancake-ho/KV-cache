#!/usr/bin/env bash
# Compatibility filename: this now checks EXISTING lab; no clone/install/repair.
set -euo pipefail
REPO_ROOT="/data/surt321/repos/lab/kv_cache"
FIGURE2_DIR="${REPO_ROOT}/baseline/crosskv/jobs/figure2"
source "${FIGURE2_DIR}/lib.sh"

fig2_check_env
printf 'Ready: FIG2_ENV=%s (existing environment; no packages changed)\n' "${FIG2_ENV}"
