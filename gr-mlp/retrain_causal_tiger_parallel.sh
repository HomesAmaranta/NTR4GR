#!/usr/bin/env bash
# Parallel next-token training for CausalTIGER.
# Each user sequence is cut into overlapping blocks of `block_items` items with
# stride `stride_items` (overlap = block_items - stride_items):
#   - length <= block_items : one block, every position supervised.
#   - length >  block_items : multiple sliding blocks; in each non-first block
#     the first `block_items - stride_items` items serve as history only
#     (masked out of the loss).
# `attention_window` is derived automatically as max_len x 4 tokens (parallel
# mode only), keeping the visible history equal to max_len items.

dataset=Beauty
block_items=${1:-40}
stride_items=${2:-20}
batch_size=${3:-256}
lr=${4:-1e-3}
lm_head=${5:-mlp}
mse_loss_weight=${6:-5}
align_target=${7:-quantized}
align_loss_type=${8:-cos}
align_item=${9:-pre}
name_suffix=${10:-}
seed=${11:-1}
log_dir=${12:-}
mse_loss_mode=${13:-mean}
early_stop_metric=ce

dataset_path="../data/${dataset}"
code_path="../data/${dataset}/${dataset}_t5_rqvae.npy"

# Auxiliary alignment loss supports token-level or mean-pooled per-item hidden
# states (parallel mode only supports item / latent / quantized targets).
case "$align_target" in
  item)
    item_emb_path="../data/${dataset}/item_emb.parquet"
    item_emb_dim=768
    ;;
  latent)
    item_emb_path="../data/${dataset}/item_emb_rqvae_encoder_latent.parquet"
    item_emb_dim=32
    ;;
  quantized)
    item_emb_path="../data/${dataset}/item_emb_rqvae_quantized_latent.parquet"
    item_emb_dim=32
    ;;
  codebook)
    item_emb_path="../data/${dataset}/item_emb_rqvae_codebook.parquet"
    item_emb_dim=32
    ;;
  *)
    echo "Unknown align_target: ${align_target}. Use item, latent, quantized, or codebook." >&2
    exit 1
    ;;
esac
case "$align_item" in
  pre|next)
    ;;
  *)
    echo "Unknown align_item: ${align_item}. Use pre or next." >&2
    exit 1
    ;;
esac
case "$mse_loss_mode" in
  mean|token)
    ;;
  *)
    echo "Unknown mse_loss_mode: ${mse_loss_mode}. Use mean or token." >&2
    exit 1
    ;;
esac

# Only encode the alignment hyper-parameters into the file name when the
# auxiliary loss is actually enabled, so plain-CE runs keep their old names.
align_suffix=""
if [ "$(awk "BEGIN{print ($mse_loss_weight > 0)}")" -eq 1 ]; then
  align_suffix="_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}_${align_target}_${align_item}"
fi

if [ -n "$name_suffix" ]; then
  name_suffix="_${name_suffix}"
fi

file_stem="causal_tiger_${dataset}_parallel_b${block_items}_s${stride_items}_bs${batch_size}_lr${lr}_head${lm_head}${align_suffix}_seed${seed}${name_suffix}"
log_dir_path="./logs"
if [ -n "$log_dir" ]; then
  log_dir_path="${log_dir_path}/${log_dir}"
fi

save_path="./ckpt/${file_stem}.pth"
log_path="${log_dir_path}/${file_stem}.log"

mkdir -p ./ckpt "$log_dir_path"
cd /mlx_devbox/users/fengyuebo/playground/TIGER/gr-mlp
/usr/bin/python main.py \
  --model_type causal \
  --train_mode parallel \
  --block_items $block_items \
  --stride_items $stride_items \
  --dataset_path $dataset_path \
  --code_path $code_path \
  --item_emb_path $item_emb_path \
  --item_emb_dim $item_emb_dim \
  --mse_loss_weight $mse_loss_weight \
  --mse_loss_mode $mse_loss_mode \
  --align_loss_type $align_loss_type \
  --align_target $align_target \
  --align_item $align_item \
  --early_stop_metric $early_stop_metric \
  --save_path $save_path \
  --log_path $log_path \
  --batch_size $batch_size \
  --infer_size 96 \
  --num_epochs 120 \
  --max_len 20 \
  --num_layers 4 \
  --num_decoder_layers 0 \
  --d_model 128 \
  --d_ff 1024 \
  --num_heads 6 \
  --d_kv 64 \
  --dropout_rate 0.1 \
  --vocab_size 1025 \
  --pad_token_id 0 \
  --eos_token_id 0 \
  --feed_forward_proj relu \
  --lm_head $lm_head \
  --lr $lr \
  --early_stop 10 \
  --beam_size 20 \
  --seed $seed
