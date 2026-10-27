#!/usr/bin/env bash
# Visualize MATO dynamic weights and cumulative differential rewards during
# decoding for a few prompts that pit two principles against each other.
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-8B}
DEVICE=${DEVICE:-cuda:0}
BATCH_INFERENCE=${BATCH_INFERENCE:-false}
SAVE_ROOT=${SAVE_ROOT:-./output/dynamics_vis}
WEIGHTS=${WEIGHTS:-"0.5 0.5"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
RHO=${RHO:-0.5}
ALPHA=${ALPHA:-0.5}

run_group () {
    local exp_name="$1"
    local p1="$2"
    local p2="$3"
    local n1="$4"
    local n2="$5"
    shift 5
    local prompts=("$@")

    python script/visualize_mato_dynamics.py \
        --model_name_or_path "${MODEL_NAME}" \
        --principles "${p1}" "${p2}" \
        --weights ${WEIGHTS} \
        --principle_names "${n1}" "${n2}" \
        --prompts "${prompts[@]}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --rho "${RHO}" \
        --alpha "${ALPHA}" \
        --do_sample true \
        --temperature 0.7 \
        --enable_thinking false \
        --batch_inference "${BATCH_INFERENCE}" \
        --dtype auto \
        --device "${DEVICE}" \
        --attn_implementation sdpa \
        --save_dir "${SAVE_ROOT}/${exp_name}"
}

# Group 1: Informativeness vs Conciseness
run_group "dynamics_info_vs_concise" \
    "Informativeness: The response should be comprehensive and include rich details, evidence, examples, and in-depth analysis to fully address the question." \
    "Conciseness: The response should be brief and to the point, avoiding unnecessary elaboration and only conveying the most essential information." \
    "Informativeness" "Conciseness" \
    "What are the pros and cons of social media?" \
    "How does climate change affect biodiversity?" \
    "What should I consider when choosing a career?"

# Group 2: Expertise vs Accessibility
run_group "dynamics_expert_vs_simple" \
    "Expertise: The response should use domain-specific terminology, assume advanced background knowledge, and provide technically precise information suited for experts." \
    "Accessibility: The response should use simple, everyday language with analogies and examples accessible to a general audience with no prior knowledge." \
    "Expertise" "Accessibility" \
    "How does the immune system recognize and fight viruses?" \
    "What is quantum entanglement and why does it matter?" \
    "How do neural networks learn from data?"

# Group 3: Positivity vs Truthfulness
run_group "dynamics_positive_vs_truth" \
    "Positivity: The response should be encouraging, optimistic, and supportive, emphasizing positive aspects and uplifting the reader's confidence." \
    "Truthfulness: The response should be factually accurate and balanced, honestly acknowledging limitations, risks, and negative aspects even when they may be discouraging." \
    "Positivity" "Truthfulness" \
    "Is it realistic to become a professional musician without formal training?" \
    "I am 45 years old. Is it too late for me to learn programming and switch careers?" \
    "Should I quit my stable job to start my own business with no savings?"

# Group 4: Formality vs Informality
run_group "dynamics_formal_vs_informal" \
    "Formality: The response should maintain a formal, professional, and objective tone, using precise language and avoiding colloquialisms or emotional expressions." \
    "Informality: The response should adopt a conversational, friendly, and casual tone, using everyday language and light humor to feel approachable and relatable." \
    "Formality" "Informality" \
    "How should I handle a conflict with my coworker?" \
    "What is the best way to apologize to a close friend after a serious argument?" \
    "How do I tell my parents I do not want to follow the career they planned for me?"

# Group 5: Evidence vs Implication
run_group "dynamics_evidence_vs_implication" \
    "Evidence: The response should ground every claim in concrete facts, statistics, studies, or verifiable data, avoiding unsupported assertions." \
    "Implication: The response should go beyond facts to explore broader implications, future consequences, and creative insights, even when speculative." \
    "Evidence" "Implication" \
    "What is the impact of artificial intelligence on employment?" \
    "How will remote work change cities and urban planning in the future?" \
    "What are the long-term consequences of antibiotic overuse?"
