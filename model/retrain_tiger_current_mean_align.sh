#!/usr/bin/env bash
# Train the original T5-based TIGER with an auxiliary representation loss.
# Default: mean-pool the final decoder layer's four target-token states and
# align them to the current item's quantized RQ-VAE latent embedding.

set -euo pipefail

dataset=${1:-Beauty}
batch_size=${2:-256}
lr=${3:-1e-4}
mse_loss_weight=${4:-1.0}
align_target=${5:-quantized}
align_loss_type=${6:-cos}
align_item=${7:-current}
seed=${8:-2025}
name_suffix=${9:-}
constrained_ce=${10:-1}
invalid_mass_loss_weight=${11:-0}

dataset_path="../data/${dataset}"
code_path="../data/${dataset}/${dataset}_t5_rqvae.npy"

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
  *)
    echo "Unknown align_target: ${align_target}. Use item, latent, or quantized." >&2
    exit 1
    ;;
esac

if [ "$align_item" = "pre" ]; then
  align_item=current
fi
case "$align_item" in
  current|next)
    ;;
  *)
    echo "Unknown align_item: ${align_item}. Use current, pre, or next." >&2
    exit 1
    ;;
esac

case "$align_loss_type" in
  mse|cos)
    ;;
  *)
    echo "Unknown align_loss_type: ${align_loss_type}. Use mse or cos." >&2
    exit 1
    ;;
esac

if [ -n "$name_suffix" ]; then
  name_suffix="_${name_suffix}"
fi

file_stem="tiger_${dataset}_${align_item}_mean_${align_target}_${align_loss_type}${mse_loss_weight}_cce${constrained_ce}_im${invalid_mass_loss_weight}_bs${batch_size}_lr${lr}_seed${seed}${name_suffix}"
ckpt_dir_path="./ckpt"
log_dir_path="./logs"
save_path="${ckpt_dir_path}/${file_stem}.pth"
log_path="${log_dir_path}/${file_stem}.log"

mkdir -p "$ckpt_dir_path" "$log_dir_path"
cd /mlx_devbox/users/fengyuebo/playground/TIGER/model

/usr/bin/python main.py \
  --dataset_path "$dataset_path" \
  --code_path "$code_path" \
  --item_emb_path "$item_emb_path" \
  --item_emb_dim "$item_emb_dim" \
  --mse_loss_weight "$mse_loss_weight" \
  --mse_loss_mode mean \
  --align_loss_type "$align_loss_type" \
  --align_target "$align_target" \
  --align_item "$align_item" \
  --constrained_ce "$constrained_ce" \
  --invalid_mass_loss_weight "$invalid_mass_loss_weight" \
  --save_path "$save_path" \
  --log_path "$log_path" \
  --batch_size "$batch_size" \
  --infer_size 96 \
  --num_epochs 200 \
  --max_len 20 \
  --num_layers 4 \
  --num_decoder_layers 4 \
  --d_model 128 \
  --d_ff 1024 \
  --num_heads 6 \
  --d_kv 64 \
  --dropout_rate 0.1 \
  --vocab_size 1025 \
  --pad_token_id 0 \
  --eos_token_id 0 \
  --feed_forward_proj relu \
  --lr "$lr" \
  --early_stop 10 \
  --beam_size 20 \
  --seed "$seed"
