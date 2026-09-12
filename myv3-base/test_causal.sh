dataset=Beauty
seed=1
align_loss_type=${1:-cos}
mse_loss_weight=${2:-5}
mse_loss_mode=${3:-mean}
align_target=${4:-quantized}
align_item=${5:-pre}
lm_head=${6:-mlp}
shallow_layer=${7:-1}
early_stop_metric=ce

dataset_path="../data/${dataset}"
code_path="../data/${dataset}/${dataset}_t5_rqvae.npy"
case "$align_target" in
  item)
    item_emb_path="../data/${dataset}/item_emb.parquet"
    item_emb_dim=768
    ;;
  latent)
    item_emb_path="../data/${dataset}/item_emb_rqvae_encoder_latent.parquet"
    item_emb_dim=32
    ;;
  quantized)
    item_emb_path="../data/${dataset}/item_emb_rqvae_quantized_latent.parquet"
    item_emb_dim=32
    ;;
  codebook)
    item_emb_path="../data/${dataset}/item_emb_rqvae_codebook.parquet"
    item_emb_dim=32
    ;;
  shallow)
    item_emb_path=None
    item_emb_dim=128
    ;;
  *)
    echo "Unknown align_target: ${align_target}. Use item, latent, quantized, codebook, or shallow." >&2
    exit 1
    ;;
esac
case "$align_item" in
  pre|next)
    ;;
  *)
    echo "Unknown align_item: ${align_item}. Use pre or next." >&2
    exit 1
    ;;
esac
case "$lm_head" in
  emb|linear|mlp)
    ;;
  *)
    echo "Unknown lm_head: ${lm_head}. Use emb or linear." >&2
    exit 1
    ;;
esac
shallow_suffix=""
if [ "$align_target" = "shallow" ]; then
  shallow_suffix="_layer${shallow_layer}"
fi
save_path="./ckpt/causal_tiger_${dataset}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}_${align_target}_${align_item}_${lm_head}${shallow_suffix}_seed${seed}.pth"
log_path="./logs/causal_tiger_${dataset}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}_${align_target}_${align_item}_${lm_head}${shallow_suffix}_seed${seed}.log"

mkdir -p ./ckpt ./logs

PROJECT_DIR="/mlx_devbox/users/fengyuebo/playground/TIGER/gr-mlp"
CHECKPOINT="${PROJECT_DIR}/ckpt/causal_tiger_Beauty_parallel_b80_s60_bs128_lr2e-3_headmlp_seed2025_0904.pth"

/usr/bin/python main.py \
  --model_type causal \
  --dataset_path $dataset_path \
  --code_path $code_path \
  --item_emb_path $item_emb_path \
  --item_emb_dim $item_emb_dim \
  --mse_loss_weight $mse_loss_weight \
  --mse_loss_mode $mse_loss_mode \
  --align_loss_type $align_loss_type \
  --align_target $align_target \
  --align_item $align_item \
  --shallow_layer $shallow_layer \
  --lm_head $lm_head \
  --early_stop_metric $early_stop_metric \
  --save_path $save_path \
  --log_path $log_path \
  --batch_size 256 \
  --infer_size 96 \
  --num_epochs 120 \
  --max_len 20 \
  --num_layers 4 \
  --num_decoder_layers 0 \
  --d_model 128 \
  --d_ff 1024 \
  --num_heads 6 \
  --d_kv 64 \
  --dropout_rate 0.1 \
  --vocab_size 1025 \
  --pad_token_id 0 \
  --eos_token_id 0 \
  --feed_forward_proj relu \
  --lr 1e-3 \
  --early_stop 10 \
  --beam_size 20 \
  --seed $seed \
  --mode evaluation \
  --save_path "${CHECKPOINT}" \

