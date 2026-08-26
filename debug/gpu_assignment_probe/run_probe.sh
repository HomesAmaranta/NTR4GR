#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-}"
SLEEP_SECONDS="${2:-0}"

if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE_PY="${PWD}/probe_gpu_assignment.py"
if [[ ! -f "${PROBE_PY}" ]]; then
  PROBE_PY="${SCRIPT_DIR}/probe_gpu_assignment.py"
fi

echo "requested_gpu_arg: ${GPU_ID:-<unset>}"
echo "effective_CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

python "${PROBE_PY}" --sleep "${SLEEP_SECONDS}"
