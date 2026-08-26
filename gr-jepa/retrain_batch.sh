#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

align_loss_types=(cos mse)
mse_loss_weights=(0.3 0.5 1)
mse_loss_modes=(token mean)
align_targets=(item latent quantized)
align_items=(pre)
lm_heads=(emb)

for align_loss_type in "${align_loss_types[@]}"; do
  for mse_loss_weight in "${mse_loss_weights[@]}"; do
    for mse_loss_mode in "${mse_loss_modes[@]}"; do
      for align_target in "${align_targets[@]}"; do
        for align_item in "${align_items[@]}"; do
          for lm_head in "${lm_heads[@]}"; do
            desc="causal_tiger_Beauty_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}_${align_target}_${align_item}_${lm_head}"
            mytask ./retrain_causal_tiger.sh \
              "$align_loss_type" \
              "$mse_loss_weight" \
              "$mse_loss_mode" \
              "$align_target" \
              "$align_item" \
              "$lm_head" \
              -m "$desc"
          done
        done
      done
    done
  done
done
