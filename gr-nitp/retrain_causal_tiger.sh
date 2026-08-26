dataset=Beauty
seed=2025
align_loss_type=cos
mse_loss_weight=0
mse_loss_mode=token
nitp_layer=4
early_stop_metric=ce

dataset_path="../data/${dataset}"
code_path="../data/${dataset}/${dataset}_t5_rqvae.npy"
save_path="./ckpt/causal_tiger_${dataset}_nitpL${nitp_layer}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}.pth"
log_path="./logs/causal_tiger_${dataset}_nitpL${nitp_layer}_${align_loss_type}${mse_loss_weight}_${mse_loss_mode}.log"

mkdir -p ./ckpt ./logs

/usr/bin/python main.py \
  --model_type causal \
  --dataset_path $dataset_path \
  --code_path $code_path \
  --mse_loss_weight $mse_loss_weight \
  --mse_loss_mode $mse_loss_mode \
  --align_loss_type $align_loss_type \
  --nitp_layer $nitp_layer \
  --early_stop_metric $early_stop_metric \
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
  --num_heads 6 \
  --d_kv 64 \
  --dropout_rate 0.1 \
  --vocab_size 1025 \
  --pad_token_id 0 \
  --eos_token_id 0 \
  --feed_forward_proj relu \
  --lr 1e-4 \
  --early_stop 10 \
  --beam_size 30 \
  --seed $seed \
  --decoder_only_lm
