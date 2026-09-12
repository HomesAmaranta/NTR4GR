#!/usr/bin/env bash
set -euo pipefail

cd /mlx_devbox/users/fengyuebo/playground/TIGER/myv3-base

python analyze_shallow_mean_bar_cos.py \
  --log_path ./logs/0908_v3_shallow/causal_tiger_Beauty_parallel_b80_s60_bs128_lr1e-3_headmlp_cos5_token_shallowL1_current_seed42.log \
  --num_sequences 100 \
  --split train \
  --device cuda
