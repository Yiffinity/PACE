#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${PROJECT_ROOT}/configs/pace_plus_msd.yaml"
LIMIT_ARGS=()
MODEL_ARGS=()
REBUILD_ARGS=()
GROUP_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --limit) LIMIT_ARGS=(--limit "$2"); shift 2 ;;
        --teacher-model) MODEL_ARGS=(--teacher-model "$2"); shift 2 ;;
        --fresh) REBUILD_ARGS=(--fresh); shift ;;
        --rebuild-from-normalizations) REBUILD_ARGS=(--rebuild-from-normalizations); shift ;;
        --group) GROUP_ARGS=(--group "$2"); shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ ${#GROUP_ARGS[@]} -eq 0 ]]; then
    echo "missing required --group: mmsd2_docmsu, mmsd2_sarcnet, or docmsu_sarcnet" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/pace_plus_cli.py" \
  --config "${CONFIG}" consolidate "${GROUP_ARGS[@]}" "${LIMIT_ARGS[@]}" "${MODEL_ARGS[@]}" "${REBUILD_ARGS[@]}"
