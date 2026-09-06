#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

block_items=80
stride_items=60
bs=128
lr=1e-3
lm_head=mlp
mse_loss_weight=5
align_loss_type=cos
log_dir=0905_cos5_item_latent

seeds=(1 2025 42)
align_targets=(codebook)
align_items=(pre next)
mse_loss_modes=(mean token)

for seed in "${seeds[@]}"; do
  for align_target in "${align_targets[@]}"; do
    for align_item in "${align_items[@]}"; do
      for mse_loss_mode in "${mse_loss_modes[@]}"; do
        desc="causal_tiger_Beauty_parallel_b${block_items}_s${stride_items}_bs${bs}_lr${lr}_head${lm_head}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}_${align_target}_${align_item}_seed${seed}"
        mytask ./retrain_causal_tiger_parallel.sh \
          "$block_items" \
          "$stride_items" \
          "$bs" \
          "$lr" \
          "$lm_head" \
          "$mse_loss_weight" \
          "$align_target" \
          "$align_loss_type" \
          "$align_item" \
          "" \
          "$seed" \
          "$log_dir" \
          "$mse_loss_mode" \
          -m "$desc"
      done
    done
  done
done
