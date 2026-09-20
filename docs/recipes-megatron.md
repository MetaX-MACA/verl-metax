# MetaX Megatron GRPO recipes

The following maintained recipes cover the validated Megatron GRPO scenarios:

- `qwen2_7b_math_megatron.sh`
- `qwen2_5_7b_megatron.sh`
- `qwen3_8b_megatron.sh`
- `qwen3_14b_megatron.sh`
- `qwen3_30b_a3b_megatron_2nodes.sh`
- `deepseek_r1_qwen_32b_megatron_2nodes.sh`
- `deepseek_r1_qwen_32b_megatron_4nodes_8k.sh`

The validated baseline is verl 0.9.0, Megatron-LM 0.18.0, vllm-metax 0.23.0,
and transformers 5.8.1. Set the checkout once for all recipes:

```bash
export MEGATRON_PATH=/workspace/Megatron-LM-0.18.0
bash examples/grpo/qwen2_5_7b_megatron.sh
```

Model and data locations have deployment defaults. To use another deployment,
set `MODEL_PATH`, `TRAIN_FILES`, and `VAL_FILES` in the shell. Extra arguments
are appended after recipe defaults, so Hydra values such as
`trainer.total_training_steps` can also be overridden at invocation time.

The 32B and Qwen3-14B recipes were validated on hosts with large CPU memory;
less than 1 TiB may cause CPU OOM when offload is enabled. Tune the offload
flags and batch sizes for the target host.

Sleep mode is not enabled by these baseline scripts. If a run needs it, first
read the [shared sleep-mode guide](../examples/env/README.md#vllm-metax-0230-sleep-mode); the
replacement is opt-in because it changes the imported vllm-metax modules in
each worker.

## Qwen3-8B dynamic context parallel

对应脚本：[`examples/grpo/qwen3_8b_megatron.sh`](../examples/grpo/qwen3_8b_megatron.sh)。

该脚本在单机 8 卡上运行 Qwen3-8B GRPO，使用 Megatron actor/reference、
vLLM rollout，并默认启用 dynamic context parallel（DCP）。

原始验证记录使用支持 `dynamic_context_parallel` 和
`max_seqlen_per_dp_cp_rank` 的 Megatron dev、megatron-bridge 0.6 和
transformers 5.x。脚本通过 `examples/env/megatron.sh` 集中加载
Megatron 路径，但 DCP 能力取决于实际 checkout。默认 checkout 不满足时，
使用 `MEGATRON_PATH` 指向兼容版本。

```bash
cd /workspace/verl-metax

MEGATRON_PATH=/path/to/Megatron-LM \
MODEL_PATH=/models/Qwen3-8B \
TRAIN_FILES=/data/gsm8k/train.parquet \
VAL_FILES=/data/gsm8k/test.parquet \
bash examples/grpo/qwen3_8b_megatron.sh
```

单机运行通常不需要提前执行 `ray start`。共享环境和 Ray 启动顺序见
[`examples/env/README.md`](../examples/env/README.md)。

脚本默认设置：

```text
actor_rollout_ref.actor.megatron.dynamic_context_parallel=True
actor_rollout_ref.actor.megatron.max_seqlen_per_dp_cp_rank=4096
```

可以通过环境变量覆盖：

```bash
DYNAMIC_CP=True \
MAX_SEQLEN_PER_DP_CP_RANK=8192 \
bash examples/grpo/qwen3_8b_megatron.sh
```

如果目标 Megatron checkout 不支持 DCP，可以使用 `DYNAMIC_CP=False` 关闭，
但这已经是不同的运行配置，需要重新评估长序列显存占用。

| 环境变量 | 默认值 |
| --- | --- |
| `ACTOR_TP` / `ACTOR_PP` | `1` / `1` |
| `ROLLOUT_TP` | `2` |
| `ROLLOUT_GPU_MEM_UTIL` | `0.15` |
| `ROLLOUT_N` | `5` |
| `MAX_RESPONSE_LENGTH` | `2048` |
| `PPO_MAX_TOKEN_LEN_PER_GPU` | `4096` |

一步冒烟测试示例：

```bash
TRAIN_BATCH_SIZE=32 \
PPO_MINI_BATCH_SIZE=32 \
bash examples/grpo/qwen3_8b_megatron.sh \
  trainer.total_training_steps=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1
```

该基线脚本默认不启用 sleep mode。如需启用，请按照
[统一 sleep mode 说明](../examples/env/README.md#vllm-metax-0230-sleep-mode)
在启动 Python 或 Ray 之前配置 replacement。
