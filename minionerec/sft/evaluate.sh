#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data/Beauty"

MODEL_PATH="${MODEL_PATH:-${1:-}}"
if [[ -z "${MODEL_PATH}" ]]; then
    echo "Set MODEL_PATH or pass the model path as the first argument." >&2
    exit 1
fi

GPU_LIST="${GPU_LIST:-0}"
RUN_NAME="$(basename "${MODEL_PATH}")"
TEMP_DIR="${PROJECT_DIR}/temp/Beauty-${RUN_NAME}"
OUTPUT_DIR="${PROJECT_DIR}/results/${RUN_NAME}"
mkdir -p "${TEMP_DIR}" "${OUTPUT_DIR}"

cd "${SCRIPT_DIR}"
python split.py \
    --input_path "${DATA_DIR}/test/Beauty.csv" \
    --output_path "${TEMP_DIR}" \
    --cuda_list "${GPU_LIST}"

for gpu in ${GPU_LIST//,/ }; do
    CUDA_VISIBLE_DEVICES="${gpu}" python -u evaluate.py \
        --base_model "${MODEL_PATH}" \
        --info_file "${DATA_DIR}/info/Beauty.txt" \
        --category Beauty \
        --test_data_path "${TEMP_DIR}/${gpu}.csv" \
        --result_json_data "${TEMP_DIR}/${gpu}.json" \
        --batch_size "${BATCH_SIZE:-8}" \
        --num_beams "${NUM_BEAMS:-50}" \
        --max_new_tokens "${MAX_NEW_TOKENS:-8}" \
        --length_penalty "${LENGTH_PENALTY:-0.0}" &
done
wait

python merge.py \
    --input_path "${TEMP_DIR}" \
    --output_path "${OUTPUT_DIR}/final_result_Beauty.json" \
    --cuda_list "${GPU_LIST}"

python calc.py \
    --path "${OUTPUT_DIR}/final_result_Beauty.json" \
    --item_path "${DATA_DIR}/info/Beauty.txt"
