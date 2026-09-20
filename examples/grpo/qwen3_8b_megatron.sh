#!/usr/bin/env bash
# GRPO | Qwen3-8B | Megatron actor/reference | vLLM rollout | 8 MetaX GPUs
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../env/megatron.sh"

MODEL_PATH=${MODEL_PATH:-/external/ai/models/llm/Qwen/Qwen3-8B}
TRAIN_FILES=${TRAIN_FILES:-/external/ai/datasets/dataset-7/ModelZoo_LLM_data/gsm8k_verl/train.parquet}
VAL_FILES=${VAL_FILES:-/external/ai/datasets/dataset-7/ModelZoo_LLM_data/gsm8k_verl/test.parquet}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
ACTOR_TP=${ACTOR_TP:-1}
ACTOR_PP=${ACTOR_PP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.15}
ROLLOUT_N=${ROLLOUT_N:-5}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF:-0}"
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP}"
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP}"
    actor_rollout_ref.actor.megatron.dynamic_context_parallel="${DYNAMIC_CP:-True}"
    actor_rollout_ref.actor.megatron.max_seqlen_per_dp_cp_rank="${MAX_SEQLEN_PER_DP_CP_RANK:-4096}"
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${ACTOR_TP}"
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${ACTOR_PP}"
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name="${PROJECT_NAME:-verl_metax_grpo}"
    trainer.experiment_name="${EXPERIMENT_NAME:-qwen3_8b_megatron_vllm}"
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.save_freq="${SAVE_FREQ:-20}"
    trainer.test_freq="${TEST_FREQ:-5}"
    trainer.total_epochs="${TOTAL_EPOCHS:-15}"
    model_engine=megatron
)

verl-metax-ppo "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${ROLLOUT[@]}" "${REF[@]}" "${TRAINER[@]}" "$@"
