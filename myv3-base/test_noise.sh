#!/usr/bin/env bash
set -euo pipefail

cd /mlx_devbox/users/fengyuebo/playground/TIGER/myv3-base

dataset=Beauty
checkpoint=${1:?Usage: bash test_noise.sh CHECKPOINT [noise_std] [noise_prob] [eval_noise] [align_loss_type] [mse_loss_weight] [mse_loss_mode] [align_target] [align_item] [lm_head] [align_hidden_mode] [seed] [log_dir]}
embedding_noise_std=${2:-1.0}
embedding_noise_prob=${3:-1.0}
eval_embedding_noise=${4:-1}
align_loss_type=${5:-cos}
mse_loss_weight=${6:-0}
mse_loss_mode=${7:-mean}
align_target=${8:-quantized}
align_item=${9:-current}
lm_head=${10:-mlp}
align_item_emb_to_hidden_mode=${11:-none}
seed=${12:-1}
log_dir=${13:-test_noise}

block_items=80
stride_items=60
batch_size=128
infer_size=96
early_stop_metric=ce
shallow_layer=1
hidden_layer=-1

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
  codebook)
    item_emb_path="../data/${dataset}/item_emb_rqvae_codebook.parquet"
    item_emb_dim=32
    ;;
  vocab|shallow)
    item_emb_path=None
    item_emb_dim=128
    ;;
  *)
    echo "Unknown align_target: ${align_target}. Use item, latent, quantized, codebook, vocab, or shallow." >&2
    exit 1
    ;;
esac

case "$align_item" in
  current|pre|next|near|nexcur)
    ;;
  *)
    echo "Unknown align_item: ${align_item}. Use current, pre, next, near, or nexcur." >&2
    exit 1
    ;;
esac
if [ "$align_item" = "nexcur" ] && { [ "$align_target" = "shallow" ] || [ "$align_target" = "vocab" ]; }; then
  echo "align_item=nexcur only supports external item embedding targets." >&2
  exit 1
fi

case "$align_item_emb_to_hidden_mode" in
  1|true|True|TRUE|yes|Yes|YES|add)
    align_item_emb_to_hidden_mode=add
    ;;
  concat)
    ;;
  0|false|False|FALSE|no|No|NO|none)
    align_item_emb_to_hidden_mode=none
    ;;
  *)
    echo "Unknown align_item_emb_to_hidden_mode: ${align_item_emb_to_hidden_mode}. Use none/add/concat or 0/1." >&2
    exit 1
    ;;
esac

eval_noise_args=()
eval_noise_suffix=""
case "$eval_embedding_noise" in
  1|true|True|TRUE|yes|Yes|YES)
    eval_noise_args=(--eval_embedding_noise)
    eval_noise_suffix="_evalnoise"
    ;;
  0|false|False|FALSE|no|No|NO)
    ;;
  *)
    echo "Unknown eval_embedding_noise: ${eval_embedding_noise}. Use 0/1 or true/false." >&2
    exit 1
    ;;
esac

align_hidden_args=()
align_hidden_suffix=""
case "$align_item_emb_to_hidden_mode" in
  add)
    align_hidden_args=(--align_item_emb_to_hidden_mode add)
    align_hidden_suffix="_addalignhidden"
    ;;
  concat)
    align_hidden_args=(--align_item_emb_to_hidden_mode concat)
    align_hidden_suffix="_concatalignhidden"
    ;;
esac

checkpoint_name=$(basename "$checkpoint")
checkpoint_stem="${checkpoint_name%.pth}"
log_dir_path="./logs/${log_dir}"
mkdir -p "$log_dir_path"
log_path="${log_dir_path}/${checkpoint_stem}_noise${embedding_noise_std}_p${embedding_noise_prob}${eval_noise_suffix}${align_hidden_suffix}_seed${seed}.log"

/usr/bin/python main.py \
  --model_type causal \
  --mode evaluation \
  --train_mode parallel \
  --block_items $block_items \
  --stride_items $stride_items \
  --dataset_path "$dataset_path" \
  --code_path "$code_path" \
  --item_emb_path "$item_emb_path" \
  --item_emb_dim $item_emb_dim \
  --mse_loss_weight $mse_loss_weight \
  --mse_loss_mode "$mse_loss_mode" \
  --align_loss_type "$align_loss_type" \
  --align_target "$align_target" \
  --align_item "$align_item" \
  --shallow_layer $shallow_layer \
  --hidden_layer $hidden_layer \
  --early_stop_metric "$early_stop_metric" \
  --save_path "$checkpoint" \
  --log_path "$log_path" \
  --batch_size $batch_size \
  --infer_size $infer_size \
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
  --embedding_noise_std $embedding_noise_std \
  --embedding_noise_prob $embedding_noise_prob \
  --lm_head "$lm_head" \
  --lr 1e-3 \
  --early_stop 10 \
  --beam_size 20 \
  --seed $seed \
  "${eval_noise_args[@]}" \
  "${align_hidden_args[@]}"
