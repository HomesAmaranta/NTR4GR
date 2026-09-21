#!/usr/bin/env bash
set -euo pipefail

cd /mlx_devbox/users/fengyuebo/playground/TIGER/myv3-base

block_items=80
stride_items=60
batch_sizes=(128)
learning_rates=(1e-3)
lm_head=mlp
align_loss_type=cos
log_dir=align

seeds=(42)
align_targets=(quantized current)
align_items=(next)
mse_loss_weights=(5)
mse_loss_modes=(mean)
shallow_layers=(1)
hidden_layers=(-1)
embedding_noise_stds=(0.0)
embedding_noise_probs=(1.0)
align_item_emb_to_hidden_modes=(none)
eval_embedding_noise_values=(0)

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
                  target_desc="${align_target}"
                  if [ "$align_target" = "shallow" ]; then
                    target_desc="shallowL${shallow_layer}"
                  fi
                  for embedding_noise_std in "${embedding_noise_stds[@]}"; do
                    for embedding_noise_prob in "${embedding_noise_probs[@]}"; do
                      for align_item_emb_to_hidden_mode in "${align_item_emb_to_hidden_modes[@]}"; do
                        for eval_embedding_noise in "${eval_embedding_noise_values[@]}"; do
                          noise_desc="_noise${embedding_noise_std}_p${embedding_noise_prob}"
                          eval_noise_desc=""
                          if [ "$eval_embedding_noise" = "1" ] || [ "$eval_embedding_noise" = "true" ]; then
                            eval_noise_desc="_evalnoise"
                          fi
                          align_hidden_desc=""
                          case "$align_item_emb_to_hidden_mode" in
                            add|1|true)
                              align_hidden_desc="_addalignhidden"
                              ;;
                            concat)
                              align_hidden_desc="_concatalignhidden"
                              ;;
                          esac
                          desc="causal_tiger_Beauty_parallel_b${block_items}_s${stride_items}_bs${bs}_lr${lr}_head${lm_head}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}_${target_desc}_${align_item}${hidden_desc}${noise_desc}${eval_noise_desc}${align_hidden_desc}_seed${seed}"
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
                            "$embedding_noise_std" \
                            "$embedding_noise_prob" \
                            "$align_item_emb_to_hidden_mode" \
                            "$eval_embedding_noise" \
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
    done
  done
done
