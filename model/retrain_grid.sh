#!/usr/bin/env bash
set -euo pipefail

cd /mlx_devbox/users/fengyuebo/playground/TIGER/model

dataset=Beauty
batch_sizes=(256)
learning_rates=(5e-4)
align_loss_type=cos

seeds=(1 42 2025)
mse_loss_weights=(0)
align_targets=(latent )
align_items=(current)

for seed in "${seeds[@]}"; do
  for align_target in "${align_targets[@]}"; do
    for align_item in "${align_items[@]}"; do
      for mse_loss_weight in "${mse_loss_weights[@]}"; do
        for bs in "${batch_sizes[@]}"; do
          for lr in "${learning_rates[@]}"; do
            desc="tiger_${dataset}_${align_item}_mean_${align_target}_${align_loss_type}${mse_loss_weight}_bs${bs}_lr${lr}_seed${seed}"
            mytask ./retrain_tiger_current_mean_align.sh \
              "$dataset" \
              "$bs" \
              "$lr" \
              "$mse_loss_weight" \
              "$align_target" \
              "$align_loss_type" \
              "$align_item" \
              "$seed" \
              "" \
              -m "$desc"
          done
        done
      done
    done
  done
done
