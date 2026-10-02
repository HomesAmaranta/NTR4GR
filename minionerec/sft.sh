#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/tiger/miniconda3/envs/MiniOneRec/bin/python
BASE_MODEL=/mlx_devbox/users/fengyuebo/playground/hf_models/Qwen3-1.7B
OUTPUT_DIR=/mnt/local/localcache00/fyb/minionerec/sft/beauty-qwen3-1.7b
DATA_DIR="${SCRIPT_DIR}/data/Beauty"
GPU_COUNT=4

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1

echo "HOST=$(hostname)"
echo "PYTHON=${PYTHON}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=0,1,2,3"
echo "GPU_COUNT=${GPU_COUNT}"
"${PYTHON}" -c "import sys, torch; print('python=', sys.executable); print('cuda=', torch.cuda.is_available(), 'count=', torch.cuda.device_count(), 'bf16=', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)"

cd "${SCRIPT_DIR}"
TORCH_DISTRIBUTED_DEBUG=DETAIL \
NCCL_NET=Socket \
NCCL_NET_PLUGIN=none \
NCCL_IB_DISABLE=1 \
NCCL_SOCKET_FAMILY=AF_INET \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
"${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc-per-node="${GPU_COUNT}" \
    --master-port=5881 \
    sft/sft.py \
    --base_model "${BASE_MODEL}" \
    --batch_size 1024 \
    --micro_batch_size 16 \
    --num_epochs 5 \
    --learning_rate 5e-4 \
    --cutoff_len 512 \
    --train_file "${DATA_DIR}/train/Beauty.csv" \
    --eval_file "${DATA_DIR}/valid/Beauty.csv" \
    --output_dir "${OUTPUT_DIR}" \
    --wandb_project "" \
    --wandb_run_name beauty-qwen3-1.7b-sft \
    --category Beauty \
    --train_from_scratch False \
    --seed 42 \
    --sid_index_path "${DATA_DIR}/index/Beauty.index.json" \
    --item_meta_path "${DATA_DIR}/index/Beauty.item.json" \
    --freeze_LLM False
