#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

block_items=80
stride_items=60
bs=128
lr=1e-3
lm_head=mlp
align_loss_type=cos
log_dir=0906_cos_shallow

seeds=(1)
align_targets=(shallow)
align_items=(near)
mse_loss_weights=(5)
mse_loss_modes=(mean)
shallow_layers=(1 2)

for seed in "${seeds[@]}"; do
  for align_target in "${align_targets[@]}"; do
    for align_item in "${align_items[@]}"; do
      for mse_loss_weight in "${mse_loss_weights[@]}"; do
        current_modes=("${mse_loss_modes[@]}")
        if [ "$align_item" = "near" ]; then
          current_modes=(token)
        fi
        for mse_loss_mode in "${current_modes[@]}"; do
          for shallow_layer in "${shallow_layers[@]}"; do
            desc="causal_tiger_Beauty_parallel_b${block_items}_s${stride_items}_bs${bs}_lr${lr}_head${lm_head}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}_${align_target}L${shallow_layer}_${align_item}_seed${seed}"
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
              "$shallow_layer" \
              -m "$desc"
          done
        done
      done
    done
  done
done
