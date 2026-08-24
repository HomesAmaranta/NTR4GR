dataset=Beauty
data=.parquet
seed=2025

save_path="./ckpt/hstu_${dataset}.pth"
log_path="./logs/hstu_${dataset}.log"

/usr/bin/python main.py \
  --type train \
  --dataset $dataset \
  --data $data \
  --save_path $save_path \
  --log_path $log_path \
  --batch_size 128 \
  --infer_size 128 \
  --num_epochs 1000 \
  --max_len 20 \
  --embedding_dim 64 \
  --num_blocks 2 \
  --num_heads 4 \
  --dqk 32 \
  --dv 32 \
  --dropout_rate 0.1 \
  --lr 1e-3 \
  --weight_decay 0 \
  --num_negatives 128 \
  --temperature 0.05 \
  --early_stop 10 \
  --seed $seed \
  --ckpt_path None
