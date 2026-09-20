#!/usr/bin/env bash
# GAE PPO | Qwen2.5-7B actor/critic | torch profiler | 8 MetaX GPUs
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../env/metax.sh"

MODEL_PATH=${MODEL_PATH:-/external/ai/models/llm/Qwen/Qwen2.5-7B-Instruct}
TRAIN_FILES=${TRAIN_FILES:-/external/ai/datasets/dataset-7/ModelZoo_LLM_data/gsm8k_verl/train.parquet}
VAL_FILES=${VAL_FILES:-/external/ai/datasets/dataset-7/ModelZoo_LLM_data/gsm8k_verl/test.parquet}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-${MODEL_PATH}}

verl-metax-ppo \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-1024}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-512}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-1024}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-256}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE:-4}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.actor.profiler.all_ranks="${PROFILE_ALL_RANKS:-False}" \
    actor_rollout_ref.actor.profiler.ranks="${PROFILE_RANKS:-[0]}" \
    actor_rollout_ref.actor.profiler.tool_config.torch.discrete="${PROFILE_DISCRETE:-False}" \
    actor_rollout_ref.actor.profiler.tool_config.torch.contents="${PROFILE_CONTENTS:-[cpu,cuda]}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-4}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.5}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.profiler.enable=True \
    actor_rollout_ref.ref.profiler.all_ranks="${PROFILE_ALL_RANKS:-False}" \
    actor_rollout_ref.ref.profiler.ranks="${PROFILE_RANKS:-[0]}" \
    actor_rollout_ref.ref.profiler.tool_config.torch.discrete="${PROFILE_DISCRETE:-False}" \
    actor_rollout_ref.ref.profiler.tool_config.torch.contents="${PROFILE_CONTENTS:-[cpu,cuda]}" \
    critic.optim.lr="${CRITIC_LR:-1e-5}" \
    critic.model.path="${CRITIC_MODEL_PATH}" \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu="${CRITIC_MICRO_BATCH_SIZE:-8}" \
    critic.profiler.enable="${PROFILE_CRITIC:-False}" \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME:-verl_metax_profile}" \
    trainer.experiment_name="${EXPERIMENT_NAME:-qwen2_5_7b_gae_torch}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE:-8}" \
    trainer.nnodes="${NNODES:-1}" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
    global_profiler.tool=torch \
    global_profiler.steps="${PROFILE_STEPS:-[2]}" \
    global_profiler.profile_continuous_steps=False \
    global_profiler.save_path="${PROFILE_DIR:-./outputs/profile_data/torch}" \
    "$@"
