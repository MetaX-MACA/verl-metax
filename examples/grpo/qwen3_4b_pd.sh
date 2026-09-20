#!/usr/bin/env bash
# GRPO | Qwen3-4B | Megatron | vLLM prefill/decode disaggregation
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../env/megatron.sh"

MODEL_PATH=${MODEL_PATH:-/metax0402/models/Qwen/Qwen3-4B}
TRAIN_FILES=${TRAIN_FILES:-/metax0402/datasets/verl/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-/metax0402/datasets/verl/gsm8k/test.parquet}

TRANSFER_BACKEND=${TRANSFER_BACKEND:-nixl}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
MAX_TOKENS_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
ACTOR_TP=${ACTOR_TP:-4}
PREFILL_TP=${PREFILL_TP:-2}
DECODE_TP=${DECODE_TP:-2}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size="${TRAIN_BATCH_SIZE:-1024}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
)

ACTOR=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-256}"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP}"
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP:-1}"
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.3}"
    actor_rollout_ref.rollout.max_model_len="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-5}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${PREFILL_TP}"
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1
    +actor_rollout_ref.rollout.enable_sleep_mode=False
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.disaggregation.enabled=True
    actor_rollout_ref.rollout.disaggregation.prefill_replicas="${PREFILL_REPLICAS:-1}"
    actor_rollout_ref.rollout.disaggregation.decode_replicas="${DECODE_REPLICAS:-3}"
    actor_rollout_ref.rollout.disaggregation.decode_tensor_model_parallel_size="${DECODE_TP}"
    actor_rollout_ref.rollout.disaggregation.transfer_backend="${TRANSFER_BACKEND}"
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}"
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${ACTOR_TP}"
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${ACTOR_PP:-1}"
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name="${PROJECT_NAME:-verl_metax_grpo}"
    trainer.experiment_name="${EXPERIMENT_NAME:-qwen3_4b_pd_${TRANSFER_BACKEND}}"
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.save_freq="${SAVE_FREQ:-20}"
    trainer.test_freq="${TEST_FREQ:-5}"
    trainer.total_epochs="${TOTAL_EPOCHS:-15}"
    model_engine=megatron
    +ray_kwargs.ray_init.object_store_memory="${RAY_OBJECT_STORE_BYTES:-107374182400}"
)

verl-metax-ppo "${DATA[@]}" "${ACTOR[@]}" "${ROLLOUT[@]}" "${REF[@]}" "${TRAINER[@]}" "$@"
