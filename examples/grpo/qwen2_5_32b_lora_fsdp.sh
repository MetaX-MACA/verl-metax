#!/usr/bin/env bash
# GRPO | Qwen2.5-32B | LoRA FSDP | vLLM rollout | 4 MetaX GPUs
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../env/metax.sh"

MODEL_PATH=${MODEL_PATH:-/pde_ai/models/llm/Qwen/Qwen2.5-32B-Instruct}
TRAIN_FILES=${TRAIN_FILES:-/pde_ai/datasets/dataset-7/ModelZoo_LLM_data/gsm8k_verl/train.parquet}
VAL_FILES=${VAL_FILES:-/pde_ai/datasets/dataset-7/ModelZoo_LLM_data/gsm8k_verl/test.parquet}

NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
ROLLOUT_TP=${ROLLOUT_TP:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

verl-metax-ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank="${LORA_RANK:-32}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA:-32}" \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-3e-5}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-128}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.6}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N:-5}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS:-512}" \
    actor_rollout_ref.rollout.max_model_len="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SEQUENCE_PARALLEL_SIZE:-2}" \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF:-0.001}" \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME:-verl_metax_grpo}" \
    trainer.experiment_name="${EXPERIMENT_NAME:-qwen2_5_32b_lora_fsdp}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES:-1}" \
    trainer.save_freq="${SAVE_FREQ:-20}" \
    trainer.test_freq="${TEST_FREQ:-20}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
    "$@"
