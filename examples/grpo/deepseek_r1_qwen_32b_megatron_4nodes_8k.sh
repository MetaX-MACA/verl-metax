#!/usr/bin/env bash
# GRPO | DeepSeek-R1-Distill-Qwen-32B | 8K response | Megatron | 4 x 8 GPUs
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export MODEL_PATH="${MODEL_PATH:-/external/models/llm/DeepSeek-R1-Distill-Qwen-32B}"
export TRAIN_FILES="${TRAIN_FILES:-/external/datasets/gsm8k_new/train.parquet}"
export VAL_FILES="${VAL_FILES:-/external/datasets/gsm8k_new/test.parquet}"
export NNODES="${NNODES:-4}"
export ACTOR_TP="${ACTOR_TP:-8}"
export ROLLOUT_TP="${ROLLOUT_TP:-8}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-2}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"

exec bash "${SCRIPT_DIR}/deepseek_r1_qwen_32b_megatron_2nodes.sh" \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size="${ACTOR_VPP:-2}" \
    trainer.experiment_name="${EXPERIMENT_NAME:-deepseek_r1_qwen_32b_megatron_4nodes_8k}" \
    "$@"

