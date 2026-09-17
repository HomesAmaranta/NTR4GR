#!/usr/bin/env bash
set -euo pipefail

cd /mlx_devbox/users/fengyuebo/playground/TIGER/abl

DATASET="${1:-${DATASET:-Beauty}}"
EMB_TYPE="${2:-${EMB_TYPE:-quantized}}"
SOURCE_OFFSET="${3:-${SOURCE_OFFSET:-0}}"
MAX_SOURCE_OFFSET="${4:-${MAX_SOURCE_OFFSET:-2}}"
SEED="${5:-${SEED:-2025}}"
BATCH_SIZE="${6:-${BATCH_SIZE:-256}}"
INFER_SIZE="${7:-${INFER_SIZE:-256}}"
LR="${8:-${LR:-1e-4}}"
HIDDEN_DIM="${9:-${HIDDEN_DIM:-128}}"
MLP_DIM="${10:-${MLP_DIM:-512}}"
DROPOUT="${11:-${DROPOUT:-0.1}}"
NUM_EPOCHS="${12:-${NUM_EPOCHS:-200}}"
EARLY_STOP="${13:-${EARLY_STOP:-10}}"
BEAM_SIZE="${14:-${BEAM_SIZE:-30}}"
TOPK_LIST="${15:-${TOPK_LIST:-5,10,20}}"
DEVICE="${16:-${DEVICE:-cuda}}"
LOG_DIR="${17:-${LOG_DIR:-logs}}"
CKPT_DIR="${18:-${CKPT_DIR:-ckpt}}"
DATA_ROOT="${DATA_ROOT:-../data/${DATASET}}"

case "${EMB_TYPE}" in
  latent)
    ITEM_EMB_PATH="${ITEM_EMB_PATH:-${DATA_ROOT}/item_emb_rqvae_encoder_latent.parquet}"
    ;;
  quantized)
    ITEM_EMB_PATH="${ITEM_EMB_PATH:-${DATA_ROOT}/item_emb_rqvae_quantized_latent.parquet}"
    ;;
  *)
    echo "Unsupported EMB_TYPE=${EMB_TYPE}; expected latent or quantized" >&2
    exit 1
    ;;
esac

RUN_NAME="${DATASET}_${EMB_TYPE}_src${SOURCE_OFFSET}_max${MAX_SOURCE_OFFSET}_seed${SEED}"

python main.py \
  --dataset_path "${DATA_ROOT}" \
  --code_path "${DATA_ROOT}/${DATASET}_t5_rqvae.npy" \
  --item_emb_path "${ITEM_EMB_PATH}" \
  --source_offset "${SOURCE_OFFSET}" \
  --max_source_offset "${MAX_SOURCE_OFFSET}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --mlp_dim "${MLP_DIM}" \
  --dropout "${DROPOUT}" \
  --batch_size "${BATCH_SIZE}" \
  --infer_size "${INFER_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --early_stop "${EARLY_STOP}" \
  --beam_size "${BEAM_SIZE}" \
  --topk_list "${TOPK_LIST}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --save_path "./${CKPT_DIR}/${RUN_NAME}.pth" \
  --log_path "./${LOG_DIR}/${RUN_NAME}.log"
