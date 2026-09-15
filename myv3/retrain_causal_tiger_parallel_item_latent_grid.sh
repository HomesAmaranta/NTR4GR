#!/usr/bin/env bash
set -euo pipefail

cd /mlx_devbox/users/fengyuebo/playground/TIGER/myv3

block_items=80
stride_items=60
batch_sizes=(128)
learning_rates=(1e-3)
lm_head=mlp
align_loss_type=cos
log_dir=0914_v3_k

seeds=(1 42 2025)
align_targets=(item latent quantized)
align_items=(current)
align_current_ks=(1)
align_current_k=(1 2 3)
mse_loss_weights=(5)
mse_loss_modes=(only-hidden)
shallow_layers=(1)
hidden_layers=(-1)

for seed in "${seeds[@]}"; do
  for align_target in "${align_targets[@]}"; do
    for align_item in "${align_items[@]}"; do
      for mse_loss_weight in "${mse_loss_weights[@]}"; do
        current_align_current_ks=(1)
        if [ "$align_item" = "current" ]; then
          if [ "${#align_current_k[@]}" -gt 0 ]; then
            current_align_current_ks=("${align_current_k[@]}")
          else
            current_align_current_ks=("${align_current_ks[@]}")
          fi
        fi
        current_modes=("${mse_loss_modes[@]}")
        if [ "$align_item" = "near" ]; then
          current_modes=(token)
        fi
        for current_k in "${current_align_current_ks[@]}"; do
          for mse_loss_mode in "${current_modes[@]}"; do
            for shallow_layer in "${shallow_layers[@]}"; do
              for hidden_layer in "${hidden_layers[@]}"; do
                for bs in "${batch_sizes[@]}"; do
                  for lr in "${learning_rates[@]}"; do
                    hidden_desc="_hiddenL${hidden_layer}"
                    target_desc="${align_target}"
                    current_k_desc=""
                    if [ "$align_target" = "shallow" ]; then
                      target_desc="shallowL${shallow_layer}"
                    elif [ "$align_item" = "current" ] && [ "$current_k" != "1" ]; then
                      current_k_desc="K${current_k}"
                    fi
                    desc="causal_tiger_Beauty_parallel_b${block_items}_s${stride_items}_bs${bs}_lr${lr}_head${lm_head}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}_${target_desc}_${align_item}${current_k_desc}${hidden_desc}_seed${seed}"
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
                      "$current_k" \
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
done
