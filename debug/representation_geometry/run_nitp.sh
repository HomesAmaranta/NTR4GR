#!/usr/bin/env bash

/usr/bin/python compute_geometry.py \
  --model_dir /mlx_devbox/users/fengyuebo/playground/TIGER/gr-nitp \
  --checkpoint_path /mlx_devbox/users/fengyuebo/playground/TIGER/gr-nitp/ckpt/causal_tiger_Beauty_nitpL5_mse0.1_token.pth \
  --dataset Beauty \
  --split train \
  --num_sequences 256 \
  --num_pairs 10000 \
  --max_len 20 \
  --nitp_layer 5 \
  --device cuda
