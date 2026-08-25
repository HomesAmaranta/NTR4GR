#!/usr/bin/env bash

/usr/bin/python compute_geometry.py \
  --model_dir /mlx_devbox/users/fengyuebo/playground/TIGER/gr \
  --checkpoint_path /mlx_devbox/users/fengyuebo/playground/TIGER/gr/ckpt/causal_tiger_Beauty.pth \
  --dataset Beauty \
  --split train \
  --num_sequences 256 \
  --num_pairs 10000 \
  --max_len 20 \
  --device cuda
