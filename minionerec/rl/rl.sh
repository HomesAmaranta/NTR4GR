#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data/Beauty"

MODEL_PATH="${MODEL_PATH:-${1:-}}"
if [[ -z "${MODEL_PATH}" ]]; then
    echo "Set MODEL_PATH or pass the SFT model path as the first argument." >&2
    exit 1
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

cd "${SCRIPT_DIR}"
accelerate launch \
    --config_file "${SCRIPT_DIR}/config/zero2_opt.yaml" \
    --num_processes "${NUM_PROCESSES:-8}" \
    --main_process_port "${MAIN_PROCESS_PORT:-29503}" \
    rl.py \
    --model_path "${MODEL_PATH}" \
    --train_batch_size "${TRAIN_BATCH_SIZE:-64}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-128}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-2}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
    --train_file "${DATA_DIR}/train/Beauty.csv" \
    --eval_file "${DATA_DIR}/valid/Beauty.csv" \
    --info_file "${DATA_DIR}/info/Beauty.txt" \
    --category Beauty \
    --sample_train False \
    --eval_step "${EVAL_STEP:-0.0999}" \
    --reward_type "${REWARD_TYPE:-ranking}" \
    --num_generations "${NUM_GENERATIONS:-16}" \
    --mask_all_zero False \
    --dynamic_sampling False \
    --sync_ref_model True \
    --beam_search True \
    --test_during_training False \
    --temperature "${TEMPERATURE:-1.0}" \
    --learning_rate "${LEARNING_RATE:-1e-5}" \
    --add_gt False \
    --beta "${BETA:-1e-3}" \
    --dapo False \
    --output_dir "${OUTPUT_DIR:-${PROJECT_DIR}/outputs/rl-beauty}" \
    --wandb_run_name "${WANDB_RUN_NAME:-rl-beauty-4level}" \
    --sid_index_path "${DATA_DIR}/index/Beauty.index.json" \
    --item_meta_path "${DATA_DIR}/index/Beauty.item.json"
