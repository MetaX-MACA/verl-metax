# verl-metax

Production-oriented MetaX hardware plugin for
[verl](https://github.com/verl-project/verl).

`verl-metax` is maintained independently from `verl` so that MetaX support can
follow the MACA, MCCL, MXSML, and `vllm-metax` release cadence.

> Status: initial release for the validated verl 0.9.0 stack. Unit tests,
> real-Ray-worker checks, single-node FSDP and Megatron training, two-node
> Megatron training, vLLM sleep mode, and the MetaX-specific runtime
> workarounds applicable to this stack have completed hardware validation. See the
> [validation matrix](docs/validation.md) for exact coverage and exclusions.

## Relationship to verl-hardware-plugin

[`verl-hardware-plugin`](https://github.com/verl-project/verl-hardware-plugin)
is the multi-vendor reference implementation for verl's hardware extension
interfaces. `verl-metax` follows those interfaces but is a separate,
MetaX-owned production package. It does not depend on `verl-hardware-plugin`.

Do not auto-load both packages for MetaX. They register the same platform name,
and vendor-specific engine registrations can conflict. Production deployments
should install `verl-metax` and select only its entry point:

```bash
export VERL_USE_EXTERNAL_PLUGINS=metax
export VERL_PLATFORM=metax
```

## Installation

Install the matching upstream verl release before installing this plugin.
verl-metax uses aligned versions: `verl-metax==0.9.0` supports exactly
`verl==0.9.0`. The complete policy for post releases and future verl versions
is documented in the [compatibility contract](docs/compatibility.md#versioning-policy).

First check whether the environment already imports a verl installation:

```bash
python -c "import verl; print(verl.__version__); print(verl.__file__)"
```

If this reports a different version, an editable checkout, or the old MetaX
verl fork, remove that installation before continuing. Also remove the old
checkout from `PYTHONPATH`; `pip uninstall` cannot remove a source directory
that is injected only through `PYTHONPATH`.

```bash
python -m pip uninstall -y verl verl-metax
python -m pip install --no-deps "verl==0.9.0"
python -m pip install --no-deps .
```

Using a local, reviewed verl 0.9.0 wheel in place of the index requirement is
also supported. The `--no-deps` form is intentional for the validated MetaX
image, whose MACA, torch, vllm-metax, NumPy, and Ray stack is provisioned
separately. Upstream verl 0.9.0 declares a NumPy requirement that does not
describe the validated NumPy `<2` MetaX environment.

Verify both versions and their import locations after installation:

```bash
python -c "import verl, verl_metax; print('verl', verl.__version__, verl.__file__); print('verl-metax', verl_metax.__version__, verl_metax.__file__)"
```

For development, install verl-metax in editable mode after the matching verl:

```bash
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

The package metadata and runtime check both fail closed outside the exact verl
compatibility boundary.

## Plugin behavior

Installing the package exposes the `metax` entry point in the `verl.plugins`
group. verl loads it automatically and registers `PlatformMetaX` as:

- device: `cuda`
- vendor: `metax`
- communication backend: `nccl` (MCCL through the CUDA-compatible PyTorch API)
- Ray resource: `GPU`
- hardware detection: `mx-smi`
- device UUID and CPU affinity provider: `pymxsml`

The first release intentionally does not register empty FSDP or Megatron
subclasses. verl's CUDA engine fallback remains active until a MetaX engine has
real behavior to override.

## Repository layout

| Path | Ownership |
| --- | --- |
| `verl_metax/` | Installable plugin, platform integration and verl 0.9.0 compatibility layer |
| `examples/` | Maintained training and profiling entry points with overridable deployment paths |
| `examples/env/` | Shared MetaX/Megatron launch environment and the authoritative sleep-mode guide |
| `docs/how-to-run/` | Full scenario runbooks, including multi-node startup procedures |
| `tools/rl_insight/` | Self-contained RL-Insight deployment, monitoring assets and training example |
| `verl_metax/sleep_mode_patch/` | Packaged, version-gated vllm-metax replacement files |

New users normally need only the compatibility contract, the relevant recipe,
and its how-to-run document. Maintainers should additionally read the
architecture and workaround policy before changing runtime behavior.

## vLLM integration

`vllm-metax` remains part of the validated runtime stack, but this plugin does
not import `vllm_metax.patch`. After installation, vllm-metax is discovered
through vLLM's plugin mechanism without an explicit import.

The versioned `vllm-run-headless` workaround remains necessary: verl 0.9.0
binds `run_headless` before vLLM plugin discovery can update the implementation,
so the workaround resolves the live function at call time.

The validated MetaX image may contain a `vllm-metax` 0.23.0 sleep-mode defect.
The installed plugin automatically loads a version-gated full-file replacement
bundle for that exact release and propagates it to child processes. See the
[sleep-mode guide](examples/env/README.md#vllm-metax-0230-sleep-mode). This is
separate from the `vllm-run-headless` verl workaround.

## Recipes and validation

Available training scenarios are listed under [`examples`](examples/README.md).
Recipes provide deployment defaults while allowing `MODEL_PATH`, `TRAIN_FILES`,
and `VAL_FILES` to override them. Validation status is tracked separately so an
available recipe is not mistaken for an end-to-end validated one; see
[`docs/validation.md`](docs/validation.md). The runbooks are organized under
[`docs/how-to-run`](docs/how-to-run/README.md), and the complete RL-Insight
deployment bundle lives under
[`tools/rl_insight`](tools/rl_insight/README.md).

See [compatibility](docs/compatibility.md),
[architecture](docs/architecture.md), and
[workaround policy](docs/workaround-policy.md).
