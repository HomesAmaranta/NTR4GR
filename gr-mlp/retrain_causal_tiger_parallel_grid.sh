#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Grid search over fixed (block_items, stride_items, batch_size, lr) tuples.

configs=(
  # "60 40 128 1e-3"
  "80 60 128 2e-3"
  # "100 80 128 2e-3"
  # "60 40 64 1e-3"
  # "80 60 64 1e-3"
  # "100 80 64 1e-3"
  # "120 100 64 1e-3"
  # "140 120 64 1e-3"
  # "160 140 64 1e-3"
)
seeds=(1 2025 42)

for config in "${configs[@]}"; do
  read -r block_items stride_items bs lr <<< "$config"
  for seed in "${seeds[@]}"; do
    desc="causal_tiger_Beauty_parallel_b${block_items}_s${stride_items}_bs${bs}_lr${lr}_seed${seed}"
    mytask ./retrain_causal_tiger_parallel.sh \
      "$block_items" \
      "$stride_items" \
      "$bs" \
      "$lr" \
      mlp \
      3 \
      quantized \
      cos \
      pre \
      "" \
      "$seed" \
      0905_cos5 \
      -m "$desc"
  done
done
