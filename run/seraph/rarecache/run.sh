#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
method="${here##*/}"
exec bash "$here/../mapper_optim/submit.sh" "$method" "$@"
