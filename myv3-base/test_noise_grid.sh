#!/usr/bin/env bash
set -euo pipefail

cd /mlx_devbox/users/fengyuebo/playground/TIGER/myv3-base

checkpoint=/mlx_devbox/users/fengyuebo/playground/TIGER/myv3-base/ckpt/base/causal_tiger_Beauty_parallel_b80_s60_bs128_lr1e-3_headmlp_noise0.0_p1.0_seed42.pth
checkpoint_name=$(basename "$checkpoint")
checkpoint_stem="${checkpoint_name%.pth}"

log_dir=0919_noise_test_last_base
seeds=(1)
embedding_noise_stds=(0.0 0.25 0.5 0.75 1.0 1.25 1.5)
embedding_noise_probs=(1.0)
eval_embedding_noise_values=(1)

align_loss_type=cos
mse_loss_weight=0
mse_loss_mode=mean
align_target=quantized
align_item=current
lm_head=mlp
align_item_emb_to_hidden_mode=none

for seed in "${seeds[@]}"; do
  for embedding_noise_std in "${embedding_noise_stds[@]}"; do
    for embedding_noise_prob in "${embedding_noise_probs[@]}"; do
      for eval_embedding_noise in "${eval_embedding_noise_values[@]}"; do
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
        desc="test_noise_${checkpoint_stem}_${align_target}_${align_item}_noise${embedding_noise_std}_p${embedding_noise_prob}${eval_noise_desc}${align_hidden_desc}_seed${seed}"
        mytask ./test_noise.sh \
          "$checkpoint" \
          "$embedding_noise_std" \
          "$embedding_noise_prob" \
          "$eval_embedding_noise" \
          "$align_loss_type" \
          "$mse_loss_weight" \
          "$mse_loss_mode" \
          "$align_target" \
          "$align_item" \
          "$lm_head" \
          "$align_item_emb_to_hidden_mode" \
          "$seed" \
          "$log_dir" \
          -m "$desc"
      done
    done
  done
done
