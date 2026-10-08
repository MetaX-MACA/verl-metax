# Validation matrix

This document records hardware validation for the verl-metax 0.9.0 release on
the [supported software stack](compatibility.md). An available
recipe is not considered validated merely because it is shipped in the
repository.

## End-to-end coverage

| Scenario | Scope | Result |
| --- | --- | --- |
| Qwen3-8B FSDP GRPO | Single node, one complete training step | Passed |
| Qwen2.5-7B Megatron GRPO | Single node, one complete training step | Passed |
| DeepSeek-R1-Distill-Qwen-32B Megatron GRPO | Two-node launch and training | Passed |
| vLLM sleep mode | Single-node and two-node training | Passed |
| Megatron ASM backend | One complete training step with `te_flash_backend=asm` | Passed |
| Fused entropy kernel | One complete training step with fused kernels and non-zero entropy coefficient | Passed |

These are smoke and compatibility runs. Their throughput and MFU values are
not release performance baselines.

## Workaround coverage

| Workaround or compatibility path | Validation | Status |
| --- | --- | --- |
| `device-flops` | Training reports finite MetaX MFU | Passed |
| `numa-affinity` | Functional affinity probe in a Ray worker | Passed |
| `vllm-run-headless` | Single-node and two-node vLLM rollout | Passed |
| `dataloader-worker-cleanup` | Normal training exit without a killed persistent DataLoader worker | Passed |
| `megatron-asm-scopes` | ASM Megatron actor forward/backward completes | Passed |
| `entropy-kernel-tile` | Worker reports the MetaX tile and fused entropy training completes | Passed |
| Ray worker propagation | `tools/verify_workarounds.py` observes process-local patches in a real worker | Passed |
| vllm-metax 0.23.0 sleep-mode replacement | Packaged auto-activation passed single-node and two-node training | Passed |
| `trtllm-device-uuid` | TensorRT-LLM is not installed in the validated stack | Not applicable / not E2E tested |

`trtllm-device-uuid` must not be described as hardware-validated until a
TensorRT-LLM deployment exercises the UUID lookup. The verification tool
reports this case as skipped when the dependency is absent.

## Revalidation commands

Run unit tests after any package, recipe, or workaround change:

```bash
python -m unittest discover -s tests -v
```

With a Ray cluster running in the correctly prepared MetaX environment, verify
that process-local patches reach a real worker:

```bash
python tools/verify_workarounds.py --address auto --num-gpus 1
```

Verify the packaged sleep-mode files independently. This command fails unless
both the driver and a real Ray worker import all three replacements from the
installed verl-metax wheel:

```bash
ray stop --force
unset RAY_ADDRESS
VERL_METAX_SLEEP_PATCH=on verl-metax-verify-sleep-patch \
  --address local --num-gpus 0
```

For a persistent multi-node cluster prepared with the same wheel and
environment on every node, replace `--address local` with `--address auto` and
add `--all-nodes`.

For single-node recipes, do not pre-start Ray unless the cluster was started
after sourcing the same MetaX environment. For multi-node recipes, source the
environment on every node before `ray start`. See the
[environment guide](../examples/env/README.md#ray-startup-order).

Each new release must repeat the focused probes and at least one complete
single-node training step. Changes to Ray propagation, vLLM, sleep mode,
Megatron, or the target verl release require rerunning their corresponding
end-to-end scenario.
