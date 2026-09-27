PYTHON=/home/tiger/miniconda3/envs/MiniOneRec/bin/python
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1 
which torchrun
which python
DATASET=Beauty
lora="--lora"
only_train_response="--only_train_response"
model_class=Qwen3-0.6B
ft=1
ckpt_name=None # your ckpt path
index_name=_t5_rqvae.npy
data_file=.parquet
post_name=test
lr=2e-5
seed=2025 # your seed
cd /mlx_devbox/users/fengyuebo/playground/TIGER/LC-Rec-backbone/scripts_qwen

model_path=/mlx_devbox/users/fengyuebo/playground/hf_models/Qwen3-0.6B
for wd in 0
do
    suffix=${model_class}-${lr}lr-${wd}wd-${suffix}
    
    OUTPUT_DIR=/mnt/local/localcache00/fyb/lcrec/${DATASET}/${suffix}
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
        --tee 3 \
        --log-dir ./torchrun_logs \
        ../finetune_lora.py \
        --base_model $model_path \
        --output_dir $OUTPUT_DIR \
        --subseq \
        --dataset $DATASET \
        --per_device_batch_size 16 \
        --gradient_accumulation_steps 2 \
        --learning_rate $lr \
        --epochs 50 \
        --lora_r 8 \
        --lora_alpha 32 \
        --lora_target_modules "q_proj,v_proj,o_proj,up_proj,down_proj" \
        --weight_decay $wd \
        --save_and_eval_strategy steps \
        --warmup_steps 200 \
        --lora_modules_to_save "embed_tokens,lm_head" \
        --index_file ${index_name} \
        --special_token_for_answer "|start_of_answer|" \
        --test_batch_size 4 \
        --resume_from_checkpoint ${ckpt_name} \
        --num_beams 20 \
        --ft 1 \
        --post $post_name \
        --seed ${seed} \
        --data_file ${data_file} \
        ${subset} \
        ${only_train_response} 
done
