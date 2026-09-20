#!/usr/bin/env bash
# GRPO | Qwen3-14B | Megatron TP4 | vLLM | 8 MetaX GPUs
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../env/megatron.sh"

MODEL_PATH=${MODEL_PATH:-/external/ai/models/llm/Qwen/Qwen3-14B}
TRAIN_FILES=${TRAIN_FILES:-/external/ai/datasets/dataset-7/ModelZoo_LLM_data/gsm8k_verl/train.parquet}
VAL_FILES=${VAL_FILES:-/external/ai/datasets/dataset-7/ModelZoo_LLM_data/gsm8k_verl/test.parquet}

verl-metax-ppo \
    --config-path=config \
    --config-name=ppo_trainer.yaml \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-64}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-1024}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-1024}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-64}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE:-1}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP:-1}" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP:-4}" \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.ignore_eos=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE:-1}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-1}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.7}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-4}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE:-32}" \
    actor_rollout_ref.ref.megatron.param_offload=True \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME:-verl_metax_gsm8k}" \
    trainer.experiment_name="${EXPERIMENT_NAME:-qwen3_14b_megatron}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE:-8}" \
    trainer.nnodes="${NNODES:-1}" \
    trainer.val_before_train=False \
    trainer.save_freq="${SAVE_FREQ:--1}" \
    trainer.test_freq="${TEST_FREQ:-10}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-5}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
    "$@"
