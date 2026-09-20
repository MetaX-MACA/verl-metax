#!/usr/bin/env bash
# GRPO | QwQ-32B | LoRA FSDP | 4 MetaX GPUs
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export MODEL_PATH="${MODEL_PATH:-/pde_ai/models/llm/Qwen/QwQ-32B}"
export TRAIN_FILES="${TRAIN_FILES:-/pde_ai/datasets/dataset-7/ModelZoo_LLM_data/gsm8k_verl/train.parquet}"
export VAL_FILES="${VAL_FILES:-/pde_ai/datasets/dataset-7/ModelZoo_LLM_data/gsm8k_verl/test.parquet}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
export SEQUENCE_PARALLEL_SIZE="${SEQUENCE_PARALLEL_SIZE:-4}"

exec bash "${SCRIPT_DIR}/qwen2_5_32b_lora_fsdp.sh" \
    data.shuffle=False \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
    actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE:-32}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_BATCHED_TOKENS:-16384}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    trainer.test_freq="${TEST_FREQ:--1}" \
    trainer.experiment_name="${EXPERIMENT_NAME:-qwq_32b_lora_fsdp}" \
    "$@"

