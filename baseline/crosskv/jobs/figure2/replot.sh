#!/usr/bin/env bash
# No model inference or fitting. Run inside an allocated compute session.

set -euo pipefail
export FIG2_RUN_DIR="${1:?Usage: replot.sh /absolute/run/directory}"
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
python -m xmodel_kv.figure2.plot --run-dir "${FIG2_RUN_DIR}"
