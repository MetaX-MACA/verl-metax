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
sleep-mode allocator and platform integration. The verl-metax wheel contains
a versioned replacement for exactly three modules without modifying the
installed vllm-metax package on disk. Importing verl loads the MetaX plugin and
activates the replacement automatically only for `vllm-metax==0.23.0`.

Enable sleep mode in the recipe; no `PYTHONPATH` or helper script is required:

```bash
bash examples/grpo/qwen2_5_7b_megatron.sh \
  +actor_rollout_ref.rollout.enable_sleep_mode=True
```

Activation is controlled by `VERL_METAX_SLEEP_PATCH`:

- `auto` (default): activate only for the validated vllm-metax 0.23.0 stack;
- `on`: require the validated version and fail if activation is unsafe;
- `off`: use the installed vllm-metax package unchanged.

Inspect the packaged redirects before training:

```bash
verl-metax-sleep-patch-info
```

Verify that the driver and a real Ray worker import all three packaged files:

```bash
ray stop --force
unset RAY_ADDRESS
VERL_METAX_SLEEP_PATCH=on verl-metax-verify-sleep-patch \
  --address local --num-gpus 0
```

Use `--skip-ray` for a driver-only check. Strict `on` mode makes an unsupported
vllm-metax version or late activation fail instead of silently skipping the
replacement. For an already running multi-node cluster, use `--address auto`
and add `--all-nodes`; install the same verl-metax wheel on every node first.

Verify the files actually imported by the driver:

```bash
python - <<'PY'
import verl
import vllm_metax.platform
from vllm_metax.device_allocator import cumem

print(vllm_metax.platform.__file__)
print(cumem.__file__)
PY
```

Both paths must point into
`verl_metax/sleep_mode_patch/_vllm_metax_0_23_0`. For a later vllm-metax
release, `auto` leaves the installed package unchanged until that stack has
been reviewed and assigned its own versioned replacement.

## Ray startup order

Single-node recipes still use Ray, but normally let `ray.init()` create a local
instance after the script has loaded the MetaX environment:

```bash
ray stop --force
unset RAY_ADDRESS
bash examples/grpo/qwen3_8b_fsdp.sh
```

For a persistent or multi-node cluster, source the required environment before
`ray start` on every node. The default `auto` mode is loaded by the installed
plugin in each worker. If overriding `VERL_METAX_SLEEP_PATCH`, export the same
value before starting Ray on every node.

```bash
source examples/env/megatron.sh
export VERL_METAX_SLEEP_PATCH=auto
ray start --head --port=1009 --num-gpus=8
```

Join worker nodes only after sourcing the same environment there. Check
`ray status` before launching training and make the driver connect to the
intended cluster explicitly when multiple local clusters may exist.
