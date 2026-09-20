#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../../examples/env/megatron.sh"

export MACA_MPS_MODE="${MACA_MPS_MODE:-1}"
export RL_INSIGHT_SERVER_URL="${RL_INSIGHT_SERVER_URL:-http://192.168.16.4:18080}"
MODEL_PATH=${MODEL_PATH:-/metax0402/m01686/models/DeepSeek-R1-Distill-Qwen-32B}
TRAIN_FILES=${TRAIN_FILES:-/metax0402/datasets/verl/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-/metax0402/datasets/verl/gsm8k/test.parquet}

verl-metax-ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml'\
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=\${data.train_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.ignore_eos=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    transfer_queue.metrics.enabled=True \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","rl_insight"]' \
    trainer.project_name='DeepSeek-R1-Distill-32B_mcore_v1' \
    trainer.experiment_name='base' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_training_steps=5 \
    trainer.total_epochs=2 \
    "$@" 2>&1 | tee train_DeepSeek-R1-Distill-Qwen-32B_4k_megatron_2nodes.log
