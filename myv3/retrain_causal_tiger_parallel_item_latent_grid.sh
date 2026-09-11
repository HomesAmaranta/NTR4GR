#!/usr/bin/env bash
set -euo pipefail

cd /mlx_devbox/users/fengyuebo/playground/TIGER/myv3

block_items=80
stride_items=60
batch_sizes=(128)
learning_rates=(3e-4  7e-4)
lm_head=mlp
align_loss_type=cos
log_dir=0908_v3_shallow_grid

seeds=(1 42 2025)
align_targets=(shallow)
align_items=(next current)
mse_loss_weights=(1 3 10)
mse_loss_modes=(mean-bar)
shallow_layers=(1 2)
hidden_layers=(-1)

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
            for hidden_layer in "${hidden_layers[@]}"; do
              for bs in "${batch_sizes[@]}"; do
                for lr in "${learning_rates[@]}"; do
                  hidden_desc=""
                  if [ "$hidden_layer" != "-1" ]; then
                    hidden_desc="_hiddenL${hidden_layer}"
                  fi
                  desc="causal_tiger_Beauty_parallel_b${block_items}_s${stride_items}_bs${bs}_lr${lr}_head${lm_head}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}_${align_target}L${shallow_layer}_${align_item}${hidden_desc}_seed${seed}"
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
                    -1 \
                    "$hidden_layer" \
                    -m "$desc"
                done
              done
            done
          done
        done
      done
    done
  done
done
