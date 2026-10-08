# Metax Megatron Muon 训练

对应脚本：[`examples/grpo/qwen2_5_7b_muon_megatron.sh`](../examples/grpo/qwen2_5_7b_muon_megatron.sh)。

该脚本基于标准 Qwen2.5-7B Megatron GRPO recipe，面向单机 8 卡训练，
为 actor 启用 layer-wise distributed Muon optimizer，同时启用 vLLM sleep
mode 和 RL-Insight 指标上报。

## 已验证环境和特殊依赖

- verl：0.9.0
- emerging-optimizers：0.4.0
- Megatron dev：`5e12dce00`
- 上述 checkout 包含 `use_layer_wise_param_layout` 所需的 upstream
  commit `3b9c248a1b`（PR #5388）

发布版 Megatron-LM 0.18.0 不包含该能力。运行 Muon recipe 时必须使用
`MEGATRON_PATH` 指向兼容的开发版 checkout；不能直接沿用其他 Megatron
recipe 的默认路径。

## 与标准 recipe 的差异

Muon 的核心配置是：

```text
actor_rollout_ref.actor.optim.optimizer=muon
actor_rollout_ref.actor.optim.use_layer_wise_distributed_optimizer=True
```

- `optimizer=muon`：通过 Megatron-Core emerging-optimizers 路径构建
  tensor-parallel-aware Muon optimizer。
- `use_layer_wise_distributed_optimizer=True`：按层分桶参与分布式优化器
  通信，降低优化器状态的显存和通信开销。

Muon 作用于 actor 的 hidden-layer 矩阵参数；embedding、head 等参数仍由
`muon_scalar_optimizer` 指定的标量优化器管理，默认使用 AdamW。

该 wrapper 还启用了 actor gradient/parameter offload、sleep mode、
transfer-queue metrics 和 RL-Insight logger。基础训练参数继续由
`qwen2_5_7b_megatron.sh` 维护。

## 运行

```bash
cd /workspace/verl-metax

MEGATRON_PATH=/path/to/Megatron-LM-dev \
MODEL_PATH=/models/Qwen2.5-7B-Instruct \
TRAIN_FILES=/data/gsm8k/train.parquet \
VAL_FILES=/data/gsm8k/test.parquet \
RL_INSIGHT_SERVER_URL=http://monitor-host:18080 \
bash examples/grpo/qwen2_5_7b_muon_megatron.sh
```

安装后的 verl-metax 插件会针对 vllm-metax 0.23.0 自动加载版本化的
sleep-mode replacement，不再需要脚本设置 `PYTHONPATH`。其激活、验证、
多机行为和移除条件统一维护在
[`examples/env/README.md`](../examples/env/README.md#vllm-metax-0230-sleep-mode)。

RL-Insight 的部署与指标说明见
[`tools/rl_insight/README.md`](../tools/rl_insight/README.md)。

## Muon 参数覆盖

verl 会把 `actor_rollout_ref.actor.optim.muon_*` 字段透传给 Megatron-Core。
常用参数包括：

```text
muon_momentum
muon_nesterov
muon_split_qkv
muon_scale_mode
muon_extra_scale_factor
muon_num_ns_steps
muon_tp_mode
muon_fp32_matmul_prec
muon_scalar_optimizer
muon_match_adamw_update_rms
```

例如：

```bash
MEGATRON_PATH=/path/to/Megatron-LM-dev \
bash examples/grpo/qwen2_5_7b_muon_megatron.sh \
  actor_rollout_ref.actor.optim.muon_momentum=0.95 \
  actor_rollout_ref.actor.optim.muon_nesterov=True \
  actor_rollout_ref.actor.optim.muon_match_adamw_update_rms=True \
  trainer.experiment_name=qwen2_5_7b_megatron_muon_v2
```

`muon_scale_mode` 和 `muon_extra_scale_factor` 会影响 update scale。
`muon_match_adamw_update_rms=True` 会按 AdamW update RMS 做对齐并叠加
`muon_extra_scale_factor`。所有 `muon_*` 参数只在 `optimizer=muon` 时生效。

## 实现入口

维护或排查 Muon 行为时，相关 verl 0.9.0 代码入口是：

- `verl/workers/config/optimizer.py`：Muon 配置字段；
- `verl/utils/megatron/optimizer.py`：Muon 参数透传和 RMS 对齐；
- `verl/workers/engine/megatron/transformer_impl.py`：layer-wise optimizer
  与 Megatron engine 的挂接点。
