#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/pace_plus_msd.yaml"
DATA="${PROJECT_ROOT}/data/pairwise_experience_splits/all_selected.jsonl"
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 "${PROJECT_ROOT}/tools/pace_plus_cli.py" --config "${CONFIG}" extract --data "${DATA}" "${EXTRA[@]}"
