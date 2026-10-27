#!/usr/bin/env bash
# Sweep MATO across (helpful, humor) preference weights on the HH-RLHF test
# split, score each run with the helpful + humor reward models, and finally
# compute hypervolume / inner-product over the resulting Pareto front.
set -euo pipefail

DATASET_NAME=${DATASET_NAME:-hh-rlhf}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-8B}
DATASET_SIZE=${DATASET_SIZE:-100}
EXP_NAME=${EXP_NAME:-mato_helpful_humor}
DEVICE=${DEVICE:-cuda:0}
BATCH_INFERENCE=${BATCH_INFERENCE:-false}
SAVE_DIR=${SAVE_DIR:-./output}
REWARD_NAMES=${REWARD_NAMES:-helpful,humor}

PRINCIPLE_HELPFUL=${PRINCIPLE_HELPFUL:-"Helpfulness: The response should provide useful resources and suggestions to the user"}
PRINCIPLE_HUMOR=${PRINCIPLE_HUMOR:-"Humour: The response should be cheerful and amusing"}

# Default weight sweep — override by exporting WEIGHTS_LIST="..." if needed.
WEIGHTS_LIST=${WEIGHTS_LIST:-"1.0 0.0;0.8 0.2;0.6 0.4;0.5 0.5;0.4 0.6;0.2 0.8;0.0 1.0"}

MODEL_SHORT=$(basename "${MODEL_NAME}")
EXP_DIR="${SAVE_DIR}/${DATASET_NAME}/${MODEL_SHORT}/mato/${EXP_NAME}"

IFS=';' read -ra WEIGHT_PAIRS <<< "${WEIGHTS_LIST}"
for W in "${WEIGHT_PAIRS[@]}"; do
    echo "=== Running MATO with weights: ${W} ==="
    python script/helpful_humor_pareto.py \
        --dataset_name "${DATASET_NAME}" \
        --dataset_size "${DATASET_SIZE}" \
        --model_name_or_path "${MODEL_NAME}" \
        --principles "${PRINCIPLE_HELPFUL}" "${PRINCIPLE_HUMOR}" \
        --weights ${W} \
        --reward_names "${REWARD_NAMES}" \
        --enable_thinking false \
        --batch_inference "${BATCH_INFERENCE}" \
        --dtype auto \
        --device "${DEVICE}" \
        --attn_implementation sdpa \
        --save_dir "${SAVE_DIR}" \
        --exp_name "${EXP_NAME}"
done

# Aggregate hypervolume / inner-product across the swept weights.
python script/multi_objective_eval.py \
    --result_path "${EXP_DIR}" \
    --reward_names "${REWARD_NAMES}"
