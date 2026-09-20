# Architecture

`verl-metax` is a sibling of `verl-hardware-plugin`, not its runtime child.

```text
verl core
  ├── PlatformRegistry / PlatformBase
  ├── EngineRegistry / BaseEngine
  └── entry point group: verl.plugins
          └── verl-metax
                ├── MetaX platform
                ├── real vendor-specific engines (only when required)
                ├── integrations: MXSML, MCCL, vllm-metax
                └── bounded, versioned workarounds
```

## Ownership boundaries

### verl core

- Hardware-neutral interfaces and call sites.
- Generic lifecycle and resource-management fixes.
- Tests proving NVIDIA and other existing platforms are unchanged.

### verl-hardware-plugin

- Multi-vendor reference implementations.
- Examples that validate extension interfaces.
- A documentation pointer to the production MetaX plugin.

### verl-metax

- MetaX platform behavior and vendor library integration.
- Validated MACA/MCCL/torch/vllm-metax compatibility.
- MetaX recipes, profiling, monitoring, and performance tuning.
- Temporary compatibility patches with explicit removal criteria.

## Engine policy

MetaX uses the `cuda` device name and the `metax` vendor name. Engine lookup can
therefore fall back to verl's CUDA implementation. A MetaX-specific engine is
registered only when it changes behavior; logging-only subclasses are avoided.
