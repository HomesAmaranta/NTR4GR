#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1

PYTHON=/home/tiger/miniconda3/envs/MiniOneRec/bin/python
GPU_COUNT=4

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "GPU_COUNT=${GPU_COUNT}"
"${PYTHON}" -c "import sys, torch; print('python=', sys.executable); print('cuda=', torch.cuda.is_available(), 'count=', torch.cuda.device_count(), 'bf16=', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)"

# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    TORCH_DISTRIBUTED_DEBUG=DETAIL \
    NCCL_DEBUG=INFO \
    NCCL_NET=Socket \
    NCCL_NET_PLUGIN=none \
    NCCL_IB_DISABLE=1 \
    NCCL_SOCKET_FAMILY=AF_INET \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    $PYTHON -m torch.distributed.run \
        --standalone \
        --nproc_per_node=4 \
        --master_port=5881 \
        sft.py \
        --base_model /mlx_devbox/users/fengyuebo/playground/hf_models/Qwen3-1.7B \
        --batch_size 1024 \
        --micro_batch_size 16 \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --output_dir /mnt/local/localcache00/fyb/minionerec/sft/0927_1.7b \
        --wandb_project wandb_proj \
        --wandb_run_name wandb_name \
        --category ${category} \
        --train_from_scratch False \
        --seed 42 \
        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
        --item_meta_path ./data/Amazon/index//Industrial_and_Scientific.item.json \
        --freeze_LLM False
done
