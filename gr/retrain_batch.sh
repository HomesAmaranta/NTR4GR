dataset=Beauty
seed=2025

dataset_path="../data/${dataset}"
code_path="../data/${dataset}/${dataset}_t5_rqvae.npy"
save_path="./ckpt/causal_tiger_${dataset}_small2.pth"
log_path="./logs/causal_tiger_${dataset}_small2.log"

mkdir -p ./ckpt ./logs

/usr/bin/python main.py \
  --model_type causal \
  --dataset_path $dataset_path \
  --code_path $code_path \
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
  --num_heads 4 \
  --d_kv 32 \
  --dropout_rate 0.1 \
  --vocab_size 1025 \
  --pad_token_id 0 \
  --eos_token_id 0 \
  --feed_forward_proj relu \
  --lr 1e-4 \
  --early_stop 10 \
  --beam_size 30 \
  --lr 1e-3 \
  --seed $seed 