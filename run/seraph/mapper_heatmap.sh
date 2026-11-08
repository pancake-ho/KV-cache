#!/usr/bin/env bash
# CPU-only post-processing entry point for the active Python environment.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd -- "$repo_root"
exec python -B -m proposed.src.exp.mapper.plot_training_heatmap "$@"
