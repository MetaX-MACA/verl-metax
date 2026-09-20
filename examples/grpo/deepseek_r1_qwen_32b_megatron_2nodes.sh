#!/usr/bin/env bash
# GRPO | DeepSeek-R1-Distill-Qwen-32B | Megatron PP2/TP4 | 2 x 8 MetaX GPUs
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../env/megatron.sh"

MODEL_PATH=${MODEL_PATH:-/external/models/llm/DeepSeek-R1-Distill-Qwen-32B}
TRAIN_FILES=${TRAIN_FILES:-/external/datasets/gsm8k_new/train.parquet}
VAL_FILES=${VAL_FILES:-/external/datasets/gsm8k_new/test.parquet}

NNODES=${NNODES:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
# 1536 is the validated OOM-safe response length for this topology.
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1536}

verl-metax-ppo \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-16}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-16}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP:-2}" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP:-4}" \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.ignore_eos="${IGNORE_EOS:-True}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE:-4}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-4}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.65}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-16}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS:-256}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_BATCHED_TOKENS:-32768}" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-0.6}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE:-16}" \
    actor_rollout_ref.ref.megatron.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME:-verl_metax_grpo}" \
    trainer.experiment_name="${EXPERIMENT_NAME:-deepseek_r1_qwen_32b_megatron_2nodes}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.save_freq="${SAVE_FREQ:--1}" \
    trainer.test_freq="${TEST_FREQ:-10}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-5}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-2}" \
    "$@"
