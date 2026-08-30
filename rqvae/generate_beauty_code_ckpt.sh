#!/usr/bin/env bash
set -euo pipefail

ckpt_name="${1:?usage: bash generate_beauty_code_ckpt.sh <ckpt_name> <output_name>}"
output_name="${2:?usage: bash generate_beauty_code_ckpt.sh <ckpt_name> <output_name>}"

cd /mlx_devbox/users/fengyuebo/playground/TIGER/rqvae

python generate_code.py \
  --dataset Beauty \
  --ckpt_path "./ckpt/Beauty/Jun-17-2025_15-21-52/${ckpt_name}" \
  --output_file "../data/Beauty/${output_name}"
