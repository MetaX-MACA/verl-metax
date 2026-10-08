#!/usr/bin/env bash
# GRPO | Qwen2.5-7B | Megatron layer-wise Muon | sleep mode | RL-Insight
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export MODEL_PATH="${MODEL_PATH:-/metax0402/metax0402/models/Qwen/Qwen2.5-7B-Instruct}"
export TRAIN_FILES="${TRAIN_FILES:-/metax0402/datasets/verl/gsm8k/train.parquet}"
export VAL_FILES="${VAL_FILES:-/metax0402/datasets/verl/gsm8k/test.parquet}"
export RL_INSIGHT_SERVER_URL="${RL_INSIGHT_SERVER_URL:-http://192.168.16.4:18080}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-2}"

exec bash "${SCRIPT_DIR}/qwen2_5_7b_megatron.sh" \
    actor_rollout_ref.actor.optim.optimizer=muon \
    actor_rollout_ref.actor.optim.use_layer_wise_distributed_optimizer=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.disable_log_stats=False \
    transfer_queue.metrics.enabled=True \
    trainer.logger='["console","rl_insight"]' \
    trainer.experiment_name="${EXPERIMENT_NAME:-qwen2_5_7b_megatron_muon}" \
    "$@"
