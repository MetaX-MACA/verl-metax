# MetaX launch environments

This directory is the authoritative reference for environment setup shared by
the training recipes. Recipe-specific runbooks should link here instead of
duplicating these instructions.

## Shared environment

`metax.sh` configures the MACA runtime defaults validated for verl-metax 0.9.0
and MACA 3.8.0. Most recipes source it indirectly. Site-specific device
visibility, network interfaces, and collective settings remain deployment
responsibilities. The stable filename is intentional: a new verl-metax release
updates the validated versions inside this file, while the Git release tag
preserves the previous environment.

`megatron.sh` sources the shared MetaX environment and adds an overridable
Megatron-LM checkout:

```bash
export MEGATRON_PATH=/path/to/Megatron-LM
source examples/env/megatron.sh
```

The default is the validated Megatron-LM 0.18.0 path. The Muon recipe requires
the development checkout documented in
[`docs/recipes-muon.md`](../../docs/recipes-muon.md).

## vllm-metax 0.23.0 sleep mode

The validated image's installed vllm-metax 0.23.0 may contain defects in its
sleep-mode allocator and platform integration. `vllm-metax-0.23-sleep-mode.sh`
enables a versioned replacement for exactly three modules without modifying
the installed environment on disk.

Enable it before starting Python or Ray:

```bash
source examples/env/vllm-metax-0.23-sleep-mode.sh
bash examples/grpo/qwen2_5_7b_muon_megatron.sh
```

The Muon recipe already sources the helper. For another recipe, source the
helper and enable the corresponding rollout option:

```bash
source examples/env/vllm-metax-0.23-sleep-mode.sh
bash examples/grpo/qwen2_5_7b_megatron.sh \
  +actor_rollout_ref.rollout.enable_sleep_mode=True
```

If the repository is mounted at a different path on worker nodes, the patch
tree must be available at the same path on every node:

```bash
export VLLM_METAX_PATCH_DIR=/opt/verl-metax/patches/vllm_metax_0_23_0
source examples/env/vllm-metax-0.23-sleep-mode.sh
```

Verify activation before training:

```bash
python - <<'PY'
import vllm_metax.platform
from vllm_metax.device_allocator import cumem

print(vllm_metax.platform.__file__)
print(cumem.__file__)
PY
```

Both paths must point into `patches/vllm_metax_0_23_0`. The implementation and
replacement file list are documented in
[`patches/vllm_metax_0_23_0`](../../patches/vllm_metax_0_23_0/README.md).

Do not enable this replacement for another vllm-metax version without comparing
all three files and creating a separately versioned patch directory. Stop
sourcing the helper when sleep mode is disabled or the installed vllm-metax
release contains equivalent fixes.

## Ray startup order

Single-node recipes still use Ray, but normally let `ray.init()` create a local
instance after the script has loaded the MetaX environment:

```bash
ray stop --force
unset RAY_ADDRESS
bash examples/grpo/qwen3_8b_fsdp.sh
```

For a persistent or multi-node cluster, source the required environment before
`ray start` on every node. Existing raylets and workers do not automatically
inherit variables set later by the training driver.

```bash
source examples/env/megatron.sh
# If required by the recipe:
source examples/env/vllm-metax-0.23-sleep-mode.sh
ray start --head --port=1009 --num-gpus=8
```

Join worker nodes only after sourcing the same environment there. Check
`ray status` before launching training and make the driver connect to the
intended cluster explicitly when multiple local clusters may exist.
