#!/usr/bin/env bash

/usr/bin/python compute_alignment_losses.py \
  --checkpoint_path /mlx_devbox/users/fengyuebo/playground/TIGER/gr/ckpt/causal_tiger_Beauty.pth \
  --dataset Beauty \
  --split valid \
  --num_samples 10 \
  --max_len 20 \
  --device cuda \
  --mse_loss_mode mean
