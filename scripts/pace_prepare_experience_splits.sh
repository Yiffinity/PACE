#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "${PROJECT_ROOT}/tools/prepare_pairwise_experience_splits.py" "$@"
