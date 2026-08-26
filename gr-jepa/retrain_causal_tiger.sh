dataset=Beauty
seed=2025
align_loss_type=${1:-cos}
mse_loss_weight=${2:-0}
mse_loss_mode=${3:-token}
early_stop_metric=ce

dataset_path="../data/${dataset}"
code_path="../data/${dataset}/${dataset}_t5_rqvae.npy"
item_emb_path="../data/${dataset}/item_emb.parquet"
save_path="./ckpt/causal_tiger_${dataset}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}.pth"
log_path="./logs/causal_tiger_${dataset}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}.log"

mkdir -p ./ckpt ./logs

/usr/bin/python main.py \
  --model_type causal \
  --dataset_path $dataset_path \
  --code_path $code_path \
  --item_emb_path $item_emb_path \
  --item_emb_dim 768 \
  --mse_loss_weight $mse_loss_weight \
  --mse_loss_mode $mse_loss_mode \
  --align_loss_type $align_loss_type \
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
  --beam_size 30 \
  --seed $seed
