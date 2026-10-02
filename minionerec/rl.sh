#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/tiger/miniconda3/envs/MiniOneRec/bin/python
MODEL_PATH=/mnt/local/localcache00/fyb/minionerec/sft/beauty-qwen3-1.7b/final_checkpoint
OUTPUT_DIR=/mnt/local/localcache00/fyb/minionerec/rl/beauty-qwen3-1.7b
DATA_DIR="${SCRIPT_DIR}/data/Beauty"
GPU_COUNT=4

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1

echo "HOST=$(hostname)"
echo "PYTHON=${PYTHON}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=0,1,2,3"
echo "GPU_COUNT=${GPU_COUNT}"
"${PYTHON}" -c "import sys, torch; print('python=', sys.executable); print('cuda=', torch.cuda.is_available(), 'count=', torch.cuda.device_count(), 'bf16=', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)"

cd "${SCRIPT_DIR}"
TORCH_DISTRIBUTED_DEBUG=DETAIL \
NCCL_DEBUG=INFO \
NCCL_NET=Socket \
NCCL_NET_PLUGIN=none \
NCCL_IB_DISABLE=1 \
NCCL_SOCKET_FAMILY=AF_INET \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
"${PYTHON}" -m accelerate.commands.launch \
    --config_file "${SCRIPT_DIR}/rl/config/zero2_opt.yaml" \
    --num_processes "${GPU_COUNT}" \
    --main_process_port 29503 \
    rl/rl.py \
    --model_path "${MODEL_PATH}" \
    --train_batch_size 64 \
    --eval_batch_size 128 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 2 \
    --train_file "${DATA_DIR}/train/Beauty.csv" \
    --eval_file "${DATA_DIR}/valid/Beauty.csv" \
    --info_file "${DATA_DIR}/info/Beauty.txt" \
    --category Beauty \
    --sample_train False \
    --eval_step 0.0999 \
    --reward_type ranking \
    --num_generations 16 \
    --mask_all_zero False \
    --dynamic_sampling False \
    --sync_ref_model True \
    --beam_search True \
    --test_during_training False \
    --temperature 1.0 \
    --learning_rate 1e-5 \
    --add_gt False \
    --beta 1e-3 \
    --dapo False \
    --output_dir "${OUTPUT_DIR}" \
    --wandb_run_name beauty-qwen3-1.7b-rl \
    --sid_index_path "${DATA_DIR}/index/Beauty.index.json" \
    --item_meta_path "${DATA_DIR}/index/Beauty.item.json"
