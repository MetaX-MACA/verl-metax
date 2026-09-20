# Metax prefill/decode disaggregation

对应脚本：[`examples/grpo/qwen3_4b_pd.sh`](../examples/grpo/qwen3_4b_pd.sh)。

该脚本在单机 8 卡上运行 Qwen3-4B GRPO，使用 Megatron actor/reference 和
vLLM prefill/decode disaggregation。默认拓扑为 1 个 prefill replica 和
3 个 decode replica，传输后端可选择 NIXL 或 Mooncake。

## 已验证环境

- verl：0.9.0
- Megatron-LM：0.18.0
- vLLM / vllm-metax：0.23.0
- 默认资源：单节点 8 卡

完整软件约束见[兼容性说明](compatibility.md)。NIXL 或 Mooncake 所需的
网络和传输依赖需要由部署环境提供。

## 运行

脚本保留了已验证环境中的模型和 GSM8K 数据路径作为默认值，也可以直接覆盖：

```bash
cd /workspace/verl-metax

MODEL_PATH=/models/Qwen3-4B \
TRAIN_FILES=/data/gsm8k/train.parquet \
VAL_FILES=/data/gsm8k/test.parquet \
TRANSFER_BACKEND=nixl \
bash examples/grpo/qwen3_4b_pd.sh
```

单机运行通常不需要提前执行 `ray start`。如需复用已有 Ray 集群，请先阅读
[Ray 启动顺序](../examples/env/README.md#ray-startup-order)。

## 拓扑参数

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `TRANSFER_BACKEND` | `nixl` | PD 数据传输后端，可设为 `nixl` 或 `mooncake` |
| `PREFILL_REPLICAS` | `1` | prefill replica 数量 |
| `DECODE_REPLICAS` | `3` | decode replica 数量 |
| `PREFILL_TP` | `2` | prefill tensor parallel size |
| `DECODE_TP` | `2` | decode tensor parallel size |
| `ACTOR_TP` | `4` | Megatron actor/reference tensor parallel size |
| `NGPUS_PER_NODE` | `8` | 每节点设备数 |
| `RAY_OBJECT_STORE_BYTES` | `107374182400` | Ray object store 大小 |

其他训练参数可以作为 Hydra 参数追加在脚本命令之后。例如，一步冒烟测试：

```bash
TRAIN_BATCH_SIZE=32 \
bash examples/grpo/qwen3_4b_pd.sh \
  trainer.total_training_steps=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1
```

## Sleep mode 限制

该场景当前显式设置：

```text
actor_rollout_ref.rollout.enable_sleep_mode=False
actor_rollout_ref.rollout.free_cache_engine=False
```

验证发现，sleep mode 使用的 CuMem/VMM allocation 不能通过当前 NIXL/UCX
IPC 路径正常导出，会导致内存注册失败。因此这里使用普通 torch caching
allocator，并同时关闭 cache-engine 释放。这个限制只属于 PD 场景；普通
vLLM rollout 的 sleep mode 已完成单双机验证。升级 vLLM、vllm-metax 或传输
后端后，应重新验证该限制，而不是直接删除这两个参数。
