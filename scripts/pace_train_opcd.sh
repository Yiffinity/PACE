#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/pace_msd.yaml"
DATA=""
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done
[[ -n "${DATA}" ]] || { echo "--data is required" >&2; exit 2; }
export PYTHONPATH="${PROJECT_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 "${PROJECT_ROOT}/tools/pace_cli.py" --config "${CONFIG}" train --data "${DATA}" "${EXTRA[@]}"
