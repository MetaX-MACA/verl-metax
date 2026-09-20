#!/usr/bin/env bash
# GRPO | DeepSeek-R1-Distill-Qwen-32B | LoRA FSDP | 4 MetaX GPUs
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export MODEL_PATH="${MODEL_PATH:-/pde_ai/models/llm/DeepSeek/DeepSeek-R1-Distill-Qwen-32B}"
export TRAIN_FILES="${TRAIN_FILES:-/pde_ai/datasets/dataset-7/ModelZoo_LLM_data/gsm8k_verl/train.parquet}"
export VAL_FILES="${VAL_FILES:-/pde_ai/datasets/dataset-7/ModelZoo_LLM_data/gsm8k_verl/test.parquet}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
export SEQUENCE_PARALLEL_SIZE="${SEQUENCE_PARALLEL_SIZE:-4}"

exec bash "${SCRIPT_DIR}/qwen2_5_32b_lora_fsdp.sh" \
    data.shuffle=False \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-0.95}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_BATCHED_TOKENS:-16384}" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    trainer.experiment_name="${EXPERIMENT_NAME:-deepseek_r1_qwen_32b_lora_fsdp}" \
    "$@"

