#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

align_loss_types=(cos mse)
mse_loss_weights=(0.01 0.3 0.7)
mse_loss_modes=(token mean)

for align_loss_type in "${align_loss_types[@]}"; do
  for mse_loss_weight in "${mse_loss_weights[@]}"; do
    for mse_loss_mode in "${mse_loss_modes[@]}"; do
      desc="causal_tiger_Beauty_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}"
      mytask ./retrain_causal_tiger.sh \
        "$align_loss_type" \
        "$mse_loss_weight" \
        "$mse_loss_mode" \
        -m "$desc"
    done
  done
done
