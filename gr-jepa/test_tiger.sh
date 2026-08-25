dataset=Beauty
model_type=${1:-causal}
align_loss_type=${2:-mse}
mse_loss_mode=${3:-token}
mse_loss_weight=0.1
seed=2025

dataset_path="../data/${dataset}"
code_path="../data/${dataset}/${dataset}_t5_rqvae.npy"

if [ "$model_type" = "gpt2" ]; then
  save_path="./ckpt/gpt2_tiger_${dataset}.pth"
  log_path="./logs/test_gpt2_tiger_${dataset}.log"
  num_heads=8
  item_emb_dim=0
elif [ "$model_type" = "causal" ]; then
  save_path="./ckpt/causal_tiger_${dataset}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}.pth"
  log_path="./logs/test_causal_tiger_${dataset}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}.log"
  num_heads=6
  item_emb_dim=768
else
  save_path="./ckpt/tiger.pth"
  log_path="./logs/test_tiger_${dataset}.log"
  num_heads=6
  item_emb_dim=0
fi

mkdir -p ./logs

/usr/bin/python main.py \
  --mode evaluation \
  --model_type $model_type \
  --dataset_path $dataset_path \
  --code_path $code_path \
  --item_emb_dim $item_emb_dim \
  --mse_loss_mode $mse_loss_mode \
  --align_loss_type $align_loss_type \
  --save_path $save_path \
  --log_path $log_path \
  --batch_size 256 \
  --infer_size 96 \
  --num_epochs 200 \
  --max_len 20 \
  --num_layers 4 \
  --num_decoder_layers 4 \
  --d_model 128 \
  --d_ff 1024 \
  --num_heads $num_heads \
  --d_kv 64 \
  --dropout_rate 0.1 \
  --vocab_size 1025 \
  --pad_token_id 0 \
  --eos_token_id 0 \
  --feed_forward_proj relu \
  --lr 1e-4 \
  --early_stop 10 \
  --beam_size 30 \
  --seed $seed
