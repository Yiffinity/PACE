#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/pace_plus_msd.yaml"
DEVICES="0,1"
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --devices) DEVICES="$2"; shift 2 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

export PYTHONPATH="${PROJECT_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
exec python3 "${PROJECT_ROOT}/tools/pace_plus_cli.py" \
  --config "${CONFIG}" benchmark --devices "${DEVICES}" "${EXTRA[@]}"
