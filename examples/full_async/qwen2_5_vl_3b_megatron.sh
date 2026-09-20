#!/usr/bin/env bash
# Fully asynchronous GRPO | Qwen2.5-VL-3B | 4 training + 4 rollout GPUs
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../env/megatron.sh"

MODEL_PATH=${MODEL_PATH:-/pde_ai/models/modelscope/modelscope.hub.metax-tech.com/models/Qwen/Qwen2.5-VL-3B-Instruct}
TRAIN_FILES=${TRAIN_FILES:-/software/m01119/datasets/verl/geo3k/train.parquet}
VAL_FILES=${VAL_FILES:-/software/m01119/datasets/verl/geo3k/test.parquet}

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-4}
TRAINING_GPUS_PER_NODE=$((NGPUS_PER_NODE - ROLLOUT_GPUS_PER_NODE))
if ((TRAINING_GPUS_PER_NODE < 1)); then
    echo "ROLLOUT_GPUS_PER_NODE must be smaller than NGPUS_PER_NODE" >&2
    exit 2
fi

verl-metax-run verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name=fully_async_ppo_megatron_trainer.yaml \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=0 \
    data.gen_batch_size="${GEN_PROMPT_BATCH_SIZE:-1}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-1024}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-2048}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
    actor_rollout_ref.actor.optim.lr_decay_steps="${LR_DECAY_STEPS:-51200}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-128}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU:-5120}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP:-2}" \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.01}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name="${ROLLOUT_ENGINE:-vllm}" \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-1}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.7}" \
    actor_rollout_ref.rollout.n="${RESPONSES_PER_PROMPT:-4}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-32768}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_BATCHED_TOKENS:-32768}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU:-5120}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU:-5120}" \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${ACTOR_TP:-2}" \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.vanilla_mbridge=True \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME:-verl_metax_geo3k}" \
    trainer.experiment_name="${EXPERIMENT_NAME:-qwen2_5_vl_3b_fully_async}" \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.resume_mode=auto \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${TRAINING_GPUS_PER_NODE}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${ROLLOUT_GPUS_PER_NODE}" \
    rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS:-51200}" \
    async_training.staleness_threshold="${STALENESS_THRESHOLD:-0.1}" \
    async_training.trigger_parameter_sync_step="${PARAMETER_SYNC_STEPS:-4}" \
    async_training.require_batches="${REQUIRE_BATCHES:-2}" \
    async_training.partial_rollout="${PARTIAL_ROLLOUT:-True}" \
    "$@"
