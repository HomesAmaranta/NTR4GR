#!/usr/bin/env bash
set -euo pipefail

cd /mlx_devbox/users/fengyuebo/playground/TIGER/abl

dataset=Beauty
emb_types=(quantized)
source_offsets=(0 1 2)
max_source_offset=2
seeds=(1 42 2025)

batch_size=256
infer_size=256
learning_rate=1e-4
hidden_dim=128
mlp_dim=512
dropout=0.1
num_epochs=120
early_stop=10
beam_size=30
topk_list=5,10,20
device=cuda
log_dir=0916_abl_offset_grid
ckpt_dir=ckpt_0916_abl_offset_grid

for emb_type in "${emb_types[@]}"; do
  for source_offset in "${source_offsets[@]}"; do
    for seed in "${seeds[@]}"; do
      case "${source_offset}" in
        0)
          offset_desc="i_to_i1"
          ;;
        1)
          offset_desc="im1_to_i1"
          ;;
        2)
          offset_desc="im2_to_i1"
          ;;
        *)
          offset_desc="offset${source_offset}_to_i1"
          ;;
      esac

      desc="abl_${dataset}_${emb_type}_${offset_desc}_max${max_source_offset}_bs${batch_size}_lr${learning_rate}_h${hidden_dim}_mlp${mlp_dim}_seed${seed}"
      mytask ./run_ablation.sh \
        "${dataset}" \
        "${emb_type}" \
        "${source_offset}" \
        "${max_source_offset}" \
        "${seed}" \
        "${batch_size}" \
        "${infer_size}" \
        "${learning_rate}" \
        "${hidden_dim}" \
        "${mlp_dim}" \
        "${dropout}" \
        "${num_epochs}" \
        "${early_stop}" \
        "${beam_size}" \
        "${topk_list}" \
        "${device}" \
        "${log_dir}" \
        "${ckpt_dir}" \
        -m "${desc}"
    done
  done
done
