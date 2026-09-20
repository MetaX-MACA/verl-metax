# How-to-run index

本目录保留各个已验证场景的完整运行说明。训练脚本统一位于
`examples/`，均提供 `MODEL_PATH`、`TRAIN_FILES`、`VAL_FILES` 默认值，
也允许在命令行前通过同名环境变量覆盖。

| 场景 | 训练脚本 | 运行说明 |
| --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-32B，Megatron，双机 | `examples/grpo/deepseek_r1_qwen_32b_megatron_2nodes.sh` | [中文](train_DeepSeek-R1-Distill-Qwen-32B_megatron_zh.md) / [English](train_DeepSeek-R1-Distill-Qwen-32B_megatron_en.md) |
| Qwen2-7B Math，Megatron | `examples/grpo/qwen2_7b_math_megatron.sh` | [中文](train_qwen2-7b_math_megatron-8gpus_zh.md) / [English](train_qwen2-7b_math_megatron-8gpus_en.md) |
| Qwen2.5-7B，Megatron | `examples/grpo/qwen2_5_7b_megatron.sh` | [中文](train_qwen2.5-7B_megatron_zh.md) / [English](train_qwen2.5-7B_megatron_en.md) |
| Qwen3-14B，Megatron | `examples/grpo/qwen3_14b_megatron.sh` | [中文](train_qwen3-14B_megatron_zh.md) / [English](train_qwen3-14B_megatron_en.md) |
| Qwen3-8B，FSDP | `examples/grpo/qwen3_8b_fsdp.sh` | [中文](train_qwen3-8b-8gpus_zh.md) / [English](train_qwen3-8b-8gpus_en.md) |
| Qwen2.5-VL-7B，FSDP | `examples/grpo/qwen2_5_vl_7b_fsdp.sh` | [中文](train_qwen2_5_vl-7b-8gpus_zh.md) / [English](train_qwen2_5_vl-7b-8gpus_en.md) |
| Qwen2.5-VL-3B，Megatron full async | `examples/full_async/qwen2_5_vl_3b_megatron.sh` | [中文](train_qwen2_5_vl-3b-megatron-8gpus_fullasync_zh.md) / [English](train_qwen2_5_vl-3b-megatron-8gpus_fullasync_en.md) |
| Qwen3-8B，Megatron dynamic CP | `examples/grpo/qwen3_8b_megatron.sh` | [Megatron 专题](../recipes-megatron.md#qwen3-8b-dynamic-context-parallel) |
| Qwen3-4B，prefill/decode 分离 | `examples/grpo/qwen3_4b_pd.sh` | [PD 专题](../recipes-pd-disaggregation.md) |
| Qwen2.5-7B，Muon | `examples/grpo/qwen2_5_7b_muon_megatron.sh` | [专题说明](../recipes-muon.md) |
| Qwen3-8B，Torch Profiler | `examples/profiling/qwen3_8b_torch.sh` | [说明](train_qwen3-8b-8gpus_torchprofiler.md) |

RL-Insight 的安装、单机/多机监控和双机训练说明集中在
[`tools/rl_insight`](../../tools/rl_insight/README.md)。启用 vLLM
sleep mode 前请阅读
[`examples/env` 中的统一环境说明](../../examples/env/README.md#vllm-metax-0230-sleep-mode)。

文档中“设置 CPU performance”依赖具体系统的 CPU governor 工具，本仓库
不提供原环境中的 `perf.sh`；请使用部署环境已有的性能模式配置方法。
