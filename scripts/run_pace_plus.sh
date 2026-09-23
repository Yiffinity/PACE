#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${PACE_PLUS_CONFIG:-${PROJECT_ROOT}/configs/pace_plus_msd.yaml}"
TEACHER_MODEL="${PACE_PLUS_TEACHER_MODEL:-${PROJECT_ROOT}/models/Qwen3.5-9B}"
STUDENT_MODEL="${PACE_PLUS_STUDENT_MODEL:-${PROJECT_ROOT}/models/Qwen3.5-4B}"
SOURCE_DATA="${PACE_PLUS_SOURCE_DATA:-${PROJECT_ROOT}/data/pairwise_experience_splits/all_selected.jsonl}"
GROUP="${PACE_PLUS_GROUP:-mmsd2_docmsu}"
TRAIN_DATA="${PACE_PLUS_TRAIN_DATA:-${PROJECT_ROOT}/data/pairwise_experience_splits/${GROUP}.jsonl}"
STAGE="${1:-all}"

case "${GROUP}" in
  mmsd2_docmsu|mmsd2_sarcnet|docmsu_sarcnet) ;;
  *) printf 'Unknown source pair: %s\n' "${GROUP}" >&2; exit 2 ;;
esac

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PACE_PLUS_TEACHER_MODEL="${TEACHER_MODEL}"
export PACE_PLUS_STUDENT_MODEL="${STUDENT_MODEL}"

run_cli() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/pace_plus_cli.py" --config "${CONFIG}" "$@"
}

case "${STAGE}" in
  preflight)
    run_cli preflight
    ;;
  prepare)
    "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/prepare_pairwise_experience_splits.py"
    ;;
  reason)
    run_cli generate-reasonings --data "${SOURCE_DATA}"
    ;;
  extract)
    run_cli extract --data "${SOURCE_DATA}"
    ;;
  consolidate)
    for group in mmsd2_docmsu mmsd2_sarcnet docmsu_sarcnet; do
      run_cli consolidate --group "${group}"
    done
    ;;
  train)
    run_cli train --data "${TRAIN_DATA}" --group "${GROUP}"
    ;;
  all)
    run_cli preflight
    "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/prepare_pairwise_experience_splits.py"
    run_cli generate-reasonings --data "${SOURCE_DATA}"
    run_cli extract --data "${SOURCE_DATA}"
    for group in mmsd2_docmsu mmsd2_sarcnet docmsu_sarcnet; do
      run_cli consolidate --group "${group}"
    done
    run_cli train --data "${TRAIN_DATA}" --group "${GROUP}"
    ;;
  *)
    printf 'Usage: %s [preflight|prepare|reason|extract|consolidate|train|all]\n' "$0" >&2
    exit 2
    ;;
esac
