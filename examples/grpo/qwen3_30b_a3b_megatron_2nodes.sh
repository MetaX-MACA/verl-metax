#!/usr/bin/env bash
# GRPO | Qwen3-30B-A3B | distributed checkpoint | Megatron PP4/EP4 | 2 x 8 GPUs
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../env/megatron.sh"

MODEL_PATH=${MODEL_PATH:-/external/models/llm/Qwen/Qwen3-30B-A3B}
DIST_CKPT_PATH=${DIST_CKPT_PATH:-/external/models/llm/Qwen/Qwen3-30B-A3B-verlmcore}
TRAIN_FILES=${TRAIN_FILES:-/external/datasets/dataset-7/ModelZoo_LLM_data/gsm8k_verl/train.parquet}
VAL_FILES=${VAL_FILES:-/external/datasets/dataset-7/ModelZoo_LLM_data/gsm8k_verl/test.parquet}

verl-metax-ppo \
    --config-path=config \
    --config-name=ppo_megatron_trainer.yaml \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-16}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-1024}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-1024}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size='${data.train_batch_size}' \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP:-4}" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP:-1}" \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size="${ACTOR_EP:-4}" \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DIST_CKPT_PATH}" \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=12 \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-4}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.7}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-4}" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS:-128}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_BATCHED_TOKENS:-8192}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path="${DIST_CKPT_PATH}" \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME:-verl_metax_qwen3_moe}" \
    trainer.experiment_name="${EXPERIMENT_NAME:-qwen3_30b_a3b_megatron}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE:-8}" \
    trainer.nnodes="${NNODES:-2}" \
    trainer.val_before_train=False \
    trainer.save_freq="${SAVE_FREQ:--1}" \
    trainer.test_freq="${TEST_FREQ:-10}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-10}" \
    "$@"
