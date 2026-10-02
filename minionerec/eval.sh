#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/tiger/miniconda3/envs/MiniOneRec/bin/python
MODEL_PATH=/mnt/local/localcache00/fyb/minionerec/rl/beauty-qwen3-1.7b/final_checkpoint
DATA_DIR="${SCRIPT_DIR}/data/Beauty"
GPU_LIST=0,1,2,3
RUN_NAME=beauty-qwen3-1.7b-rl
TEMP_DIR="${SCRIPT_DIR}/temp/Beauty-${RUN_NAME}"
OUTPUT_DIR="${SCRIPT_DIR}/results/${RUN_NAME}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONNOUSERSITE=1

echo "HOST=$(hostname)"
echo "PYTHON=${PYTHON}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CUDA_VISIBLE_DEVICES=${GPU_LIST}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p "${TEMP_DIR}" "${OUTPUT_DIR}"
cd "${SCRIPT_DIR}"

"${PYTHON}" sft/split.py \
    --input_path "${DATA_DIR}/test/Beauty.csv" \
    --output_path "${TEMP_DIR}" \
    --cuda_list "${GPU_LIST}"

for gpu in ${GPU_LIST//,/ }; do
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u sft/evaluate.py \
        --base_model "${MODEL_PATH}" \
        --info_file "${DATA_DIR}/info/Beauty.txt" \
        --category Beauty \
        --test_data_path "${TEMP_DIR}/${gpu}.csv" \
        --result_json_data "${TEMP_DIR}/${gpu}.json" \
        --batch_size 8 \
        --num_beams 50 \
        --max_new_tokens 8 \
        --length_penalty 0.0 &
done
wait

"${PYTHON}" sft/merge.py \
    --input_path "${TEMP_DIR}" \
    --output_path "${OUTPUT_DIR}/final_result_Beauty.json" \
    --cuda_list "${GPU_LIST}"

"${PYTHON}" sft/calc.py \
    --path "${OUTPUT_DIR}/final_result_Beauty.json" \
    --item_path "${DATA_DIR}/info/Beauty.txt"
