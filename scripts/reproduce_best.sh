#!/usr/bin/env bash
# Best-results entrypoint. Dry-run by default; see --list and --help.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PY:-python3}" scripts/run_best.py "$@"
