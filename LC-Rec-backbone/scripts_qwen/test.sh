PYTHON=/home/tiger/miniconda3/envs/MiniOneRec/bin/python
export WANDB_MODE=disabled

unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset TORCH_DISTRIBUTED_DEBUG

unset NCCL_SOCKET_IFNAME
DATASET=Beauty
model_class=Qwen3-0.6B
index_name=_t5_rqvae.npy
data_file=.parquet
lr=2e-5
seed=2025
mse_loss_weight=${MSE_LOSS_WEIGHT:-0}
align_target=${ALIGN_TARGET:-latent}
align_item=${ALIGN_ITEM:-current}
align_loss_type=cos
cd /mlx_devbox/users/fengyuebo/playground/TIGER/LC-Rec-backbone/scripts_qwen

model_path=/mlx_devbox/users/fengyuebo/playground/hf_models/Qwen3-0.6B

for wd in 0
do
    suffix=${model_class}-${lr}lr-${wd}wd-${suffix}
    if [ "$mse_loss_weight" != "0" ]; then
        suffix=${suffix}-${align_item}-${align_target}-${align_loss_type}${mse_loss_weight}
    fi
    ckpt_path=${CKPT_PATH:-/mlx_devbox/users/fengyuebo/playground/TIGER/LC-Rec-backbone/ckpt/${DATASET}/${suffix}}

    TORCH_DISTRIBUTED_DEBUG=DETAIL \
    NCCL_NET=Socket \
    NCCL_NET_PLUGIN=none \
    NCCL_IB_DISABLE=1 \
    NCCL_SOCKET_FAMILY=AF_INET \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    $PYTHON -m torch.distributed.run \
        --standalone \
        --nproc_per_node=4 \
        --master_port=5881 \
        ../test_qwen_ddp.py \
        --dataset $DATASET \
        --base_model $model_path \
        --ckpt_path "$ckpt_path" \
        --test_batch_size 32 \
        --num_beams 20 \
        --seed ${seed} \
        --lora_r 8 \
        --lora_alpha 32 \
        --lora_target_modules "q_proj,v_proj,o_proj,up_proj,down_proj" \
        --lora_modules_to_save "embed_tokens,lm_head" \
        --index_file ${index_name} \
        --data_file ${data_file} \
        --special_token_for_answer "|start_of_answer|" \
        --mse_loss_weight ${mse_loss_weight} \
        --align_target ${align_target} \
        --align_item ${align_item} \
        --align_loss_type ${align_loss_type} \
        --lora
done
