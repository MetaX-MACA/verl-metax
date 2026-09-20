# MetaX multimodal recipes

Two Geo3K scenarios are retained:

| Script | Topology |
| --- | --- |
| `examples/grpo/qwen2_5_vl_7b_fsdp.sh` | colocated FSDP actor and vLLM rollout on 8 GPUs |
| `examples/full_async/qwen2_5_vl_3b_megatron.sh` | 4 Megatron training GPUs plus 4 asynchronous rollout GPUs |

Both recipes set `mm_processor_cache_gb=0` to avoid multimodal processor-cache
pressure in these runs. The FSDP recipe supports transformers 4.57.1 and the
validated 5.8.1. Validation with 5.10 and newer failed because
`grayscale_to_rgb` was unavailable, so those versions are outside this release
contract.

Override the defaults without editing either script:

```bash
MODEL_PATH=/models/Qwen2.5-VL-7B-Instruct \
TRAIN_FILES=/data/geo3k/train.parquet \
VAL_FILES=/data/geo3k/test.parquet \
bash examples/grpo/qwen2_5_vl_7b_fsdp.sh
```

For full async, `NGPUS_PER_NODE` is the total and
`ROLLOUT_GPUS_PER_NODE` defaults to 4. The script derives the number of
training GPUs and rejects a zero-sized training partition.
