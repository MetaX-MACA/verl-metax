# MetaX recipes for verl 0.9.0

These recipes are maintained starting points for verl 0.9.0 on MetaX. Each
script keeps a validated topology and exposes deployment-specific paths and
common launch settings as overrides. Availability here does not imply that
every scenario has completed end-to-end validation in the current release;
see the [validation matrix](../docs/validation.md).

| Script | Coverage |
| --- | --- |
| `grpo/qwen2_7b_math_megatron.sh` | Qwen2-7B, mixed GSM8K/MATH, Megatron |
| `grpo/qwen2_5_7b_megatron.sh` | Qwen2.5-7B, Megatron PP2 |
| `grpo/qwen2_5_7b_muon_megatron.sh` | Layer-wise Muon, sleep mode, RL-Insight |
| `grpo/qwen3_8b_megatron.sh` | Qwen3-8B Megatron with dynamic context parallelism |
| `grpo/qwen3_14b_megatron.sh` | Qwen3-14B Megatron TP4 |
| `grpo/qwen3_30b_a3b_megatron_2nodes.sh` | Qwen3 MoE, distributed checkpoint, 2 nodes |
| `grpo/qwen3_8b_fsdp.sh` | Qwen3-8B FSDP |
| `grpo/qwen3_4b_pd.sh` | vLLM prefill/decode disaggregation |
| `grpo/qwen2_5_vl_7b_fsdp.sh` | Qwen2.5-VL-7B FSDP on Geo3K |
| `grpo/qwen2_5_32b_lora_fsdp.sh` | Qwen2.5-32B LoRA FSDP |
| `grpo/deepseek_r1_qwen_32b_lora_fsdp.sh` | DeepSeek-R1-Distill-Qwen-32B LoRA FSDP |
| `grpo/qwq_32b_lora_fsdp.sh` | QwQ-32B LoRA FSDP |
| `grpo/deepseek_r1_qwen_32b_megatron_2nodes.sh` | DeepSeek 32B, PP2/TP4, 2 nodes |
| `grpo/deepseek_r1_qwen_32b_megatron_4nodes_8k.sh` | DeepSeek 32B, 8K response, 4 nodes |
| `full_async/qwen2_5_vl_3b_megatron.sh` | Fully asynchronous VLM, 4 training + 4 rollout GPUs |
| `profiling/qwen3_8b_torch.sh` | GRPO actor torch profiler smoke run |
| `profiling/qwen2_5_7b_gae_torch.sh` | GAE actor/ref/critic torch profiler run |

Every recipe has the last validated deployment paths as defaults. They are
examples, not package installation paths, and can be overridden without
editing the file:

```bash
MODEL_PATH=/models/Qwen3-8B \
TRAIN_FILES=/data/gsm8k/train.parquet \
VAL_FILES=/data/gsm8k/test.parquet \
bash examples/grpo/qwen3_8b_megatron.sh trainer.total_training_steps=5
```

`TRAIN_FILES` and `VAL_FILES` may also be Hydra list expressions. Megatron
recipes source `env/megatron.sh`, whose default checkout is
`/workspace/Megatron-LM-0.18.0`; override it with `MEGATRON_PATH`.

Shared MetaX, Megatron, sleep-mode, and Ray startup instructions are maintained
in [`env/README.md`](env/README.md).

Start with the [how-to-run index](../docs/how-to-run/README.md) for the
step-by-step runbooks. Additional focused references are available here:

- [Megatron GRPO](../docs/recipes-megatron.md)
- [VLM recipes](../docs/recipes-vlm.md)
- [Muon](../docs/recipes-muon.md)
- [RL-Insight installation and multi-node training](../tools/rl_insight/README.md)
- [PD disaggregation](../docs/recipes-pd-disaggregation.md)
- [Profiling](../docs/profiling.md)
