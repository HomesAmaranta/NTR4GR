#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

align_loss_types=(cos mse)
mse_loss_weights=(2 5)
mse_loss_modes=(token mean)
align_targets=(item latent quantized)
align_items=(pre next)
lm_heads=(emb)
shallow_layers=(1)

for align_loss_type in "${align_loss_types[@]}"; do
  for mse_loss_weight in "${mse_loss_weights[@]}"; do
    for mse_loss_mode in "${mse_loss_modes[@]}"; do
      for align_target in "${align_targets[@]}"; do
        cur_align_items=("${align_items[@]}")
        cur_shallow_layers=(1)
        if [ "$align_target" = "shallow" ]; then
          cur_align_items=(next)
          cur_shallow_layers=("${shallow_layers[@]}")
        fi
        for align_item in "${cur_align_items[@]}"; do
          for lm_head in "${lm_heads[@]}"; do
            for shallow_layer in "${cur_shallow_layers[@]}"; do
              shallow_suffix=""
              if [ "$align_target" = "shallow" ]; then
                shallow_suffix="_layer${shallow_layer}"
              fi
              desc="causal_tiger_Beauty_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}_${align_target}_${align_item}_${lm_head}${shallow_suffix}"
              mytask ./retrain_causal_tiger.sh \
                "$align_loss_type" \
                "$mse_loss_weight" \
                "$mse_loss_mode" \
                "$align_target" \
                "$align_item" \
                "$lm_head" \
                "$shallow_layer" \
                -m "$desc"
            done
          done
        done
      done
    done
  done
done
