#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data/Beauty"

BASE_MODEL="${BASE_MODEL:-${1:-}}"
if [[ -z "${BASE_MODEL}" ]]; then
    echo "Set BASE_MODEL or pass the model path as the first argument." >&2
    exit 1
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

cd "${SCRIPT_DIR}"
torchrun --nproc_per_node "${NPROC_PER_NODE:-8}" \
    sft.py \
    --base_model "${BASE_MODEL}" \
    --batch_size "${BATCH_SIZE:-1024}" \
    --micro_batch_size "${MICRO_BATCH_SIZE:-16}" \
    --num_epochs "${NUM_EPOCHS:-10}" \
    --cutoff_len "${CUTOFF_LEN:-512}" \
    --train_file "${DATA_DIR}/train/Beauty.csv" \
    --eval_file "${DATA_DIR}/valid/Beauty.csv" \
    --output_dir "${OUTPUT_DIR:-${PROJECT_DIR}/outputs/sft-beauty}" \
    --wandb_project "${WANDB_PROJECT:-}" \
    --wandb_run_name "${WANDB_RUN_NAME:-sft-beauty-4level}" \
    --category Beauty \
    --train_from_scratch False \
    --seed "${SEED:-42}" \
    --sid_index_path "${DATA_DIR}/index/Beauty.index.json" \
    --item_meta_path "${DATA_DIR}/index/Beauty.item.json" \
    --freeze_LLM False
