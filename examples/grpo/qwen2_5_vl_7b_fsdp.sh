#!/usr/bin/env bash
# GRPO | Qwen2.5-VL-7B | Geo3K | FSDP | vLLM | 8 MetaX GPUs
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../env/metax.sh"

MODEL_PATH=${MODEL_PATH:-/external/ai/models/llm/Qwen/Qwen2.5-VL-7B-Instruct}
TRAIN_FILES=${TRAIN_FILES:-/external/ai/datasets/dataset-7/ModelZoo_LLM_data/hiyouga/geometry3k/train.parquet}
VAL_FILES=${VAL_FILES:-/external/ai/datasets/dataset-7/ModelZoo_LLM_data/hiyouga/geometry3k/test.parquet}

verl-metax-ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-256}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-1024}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-2048}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.image_key="${IMAGE_KEY:-images}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-128}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE:-2}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.01}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.ignore_eos=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE:-4}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-2}" \
    actor_rollout_ref.rollout.name="${ROLLOUT_ENGINE:-vllm}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.6}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-5}" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE:-4}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME:-verl_metax_geo3k}" \
    trainer.experiment_name="${EXPERIMENT_NAME:-qwen2_5_vl_7b_fsdp}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE:-8}" \
    trainer.nnodes="${NNODES:-1}" \
    trainer.save_freq="${SAVE_FREQ:--1}" \
    trainer.test_freq="${TEST_FREQ:-5}" \
    trainer.val_before_train=False \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-5}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-15}" \
    "$@"
