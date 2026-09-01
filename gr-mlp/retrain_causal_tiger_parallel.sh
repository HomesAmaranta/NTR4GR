#!/usr/bin/env bash
# Parallel next-token training for CausalTIGER.
# One user sequence is cut into overlapping blocks of `block_items` items with
# stride `stride_items`; the causal model predicts every next token in a single
# forward pass. `attention_window` (in tokens) keeps the visible history bounded
# to 20 items (20 x 4 codes = 80).

dataset=Beauty
seed=2025
block_items=${1:-40}
stride_items=${2:-20}
attention_window=${3:-80}
early_stop_metric=ce

dataset_path="../data/${dataset}"
code_path="../data/${dataset}/${dataset}_t5_rqvae.npy"

save_path="./ckpt/causal_tiger_${dataset}_parallel_b${block_items}_s${stride_items}_w${attention_window}_seed${seed}.pth"
log_path="./logs/causal_tiger_${dataset}_parallel_b${block_items}_s${stride_items}_w${attention_window}_seed${seed}.log"

mkdir -p ./ckpt ./logs

/usr/bin/python main.py \
  --model_type causal \
  --train_mode parallel \
  --block_items $block_items \
  --stride_items $stride_items \
  --attention_window $attention_window \
  --dataset_path $dataset_path \
  --code_path $code_path \
  --early_stop_metric $early_stop_metric \
  --save_path $save_path \
  --log_path $log_path \
  --batch_size 256 \
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
  --lr 1e-3 \
  --early_stop 10 \
  --beam_size 20 \
  --seed $seed
