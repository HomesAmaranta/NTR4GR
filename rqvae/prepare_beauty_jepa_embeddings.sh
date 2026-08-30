#!/usr/bin/env bash
set -euo pipefail

cd /mlx_devbox/users/fengyuebo/playground/TIGER/rqvae

python prepare_jepa_embeddings.py \
  --item_emb_path ../data/Beauty/item_emb.parquet \
  --rqvae_ckpt ./ckpt/Beauty/Jun-17-2025_15-21-52/best_collision_model.pth \
  --output_dir ../data/Beauty \
  --target_code_len 4
