#!/usr/bin/env bash
# Run MATO on Multifaceted-Bench with integrated LLM-as-judge evaluation, then
# print the aggregate report (Helpfulness / ACS / AMR / APS / Variance / Worst).
#
# Required environment for the judge:
#   OpenAI:        OPENAI_API_KEY
#   Azure OpenAI:  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
#   Anthropic:     ANTHROPIC_API_KEY
# These can also be set in a `.env` file at the repo root.
set -euo pipefail

DATASET_NAME=${DATASET_NAME:-multifaceted}
MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}
DATASET_SIZE=${DATASET_SIZE:--1}            # -1 (or unset) means use the full split
EXP_NAME=${EXP_NAME:-mato-${DATASET_SIZE}}
JUDGE_PROVIDER=${JUDGE_PROVIDER:-openai}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-4o}
JUDGE_NUM_WORKERS=${JUDGE_NUM_WORKERS:-4}
DEVICE=${DEVICE:-cuda:0}
BATCH_INFERENCE=${BATCH_INFERENCE:-false}
SAVE_DIR=${SAVE_DIR:-./output}

MODEL_SHORT=$(basename "${MODEL_NAME}")

python script/personalized_reasoning/multifaceted_inference.py \
    --dataset_name "${DATASET_NAME}" \
    --dataset_size "${DATASET_SIZE}" \
    --model_name_or_path "${MODEL_NAME}" \
    --enable_thinking false \
    --batch_inference "${BATCH_INFERENCE}" \
    --dtype auto \
    --device "${DEVICE}" \
    --attn_implementation sdpa \
    --exp_name "${EXP_NAME}" \
    --save_dir "${SAVE_DIR}" \
    --run_judge true \
    --judge_provider "${JUDGE_PROVIDER}" \
    --judge_model "${JUDGE_MODEL}" \
    --judge_num_workers "${JUDGE_NUM_WORKERS}"

python script/personalized_reasoning/multifaceted_report.py \
    --judge_model "${JUDGE_MODEL}" \
    --eval_dir "${SAVE_DIR}/${DATASET_NAME}/${MODEL_SHORT}/mato/${EXP_NAME}/judge_eval_results.${JUDGE_MODEL}.json"
