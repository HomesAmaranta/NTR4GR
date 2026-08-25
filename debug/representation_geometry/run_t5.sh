#!/usr/bin/env bash

/usr/bin/python compute_geometry.py \
  --architecture t5 \
  --model_dir /mlx_devbox/users/fengyuebo/playground/TIGER/model \
  --checkpoint_path /mlx_devbox/users/fengyuebo/playground/TIGER/model/ckpt/tiger.pth \
  --dataset Beauty \
  --split train \
  --num_sequences 256 \
  --num_pairs 10000 \
  --max_len 20 \
  --device cuda
