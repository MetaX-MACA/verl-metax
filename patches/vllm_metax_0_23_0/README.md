# vllm-metax 0.23.0 sleep-mode replacement

This directory contains the Python modules required to enable vLLM sleep mode
in the validated MetaX 3.8 runtime. It is intentionally version-specific and
must only be used with the matching `vllm==0.23.0` and `vllm-metax` stack.

The image's installed `vllm_metax` package may contain sleep-mode defects that
cannot be repaired by changing verl alone. The replacement bundle supplies:

- `vllm_metax/platform.py`;
- `vllm_metax/device_allocator/cumem.py`;
- `vllm_metax/patch/plugin_enhancement/device_allocator.py`.

`sitecustomize.py` installs a Python import finder at interpreter startup. It
loads the three files above instead of their installed-package counterparts;
all other `vllm_metax` modules continue to come from the installed package.
No files in the Python environment are modified on disk.

## Activation

Operator-facing activation, verification, multi-node ordering, and removal
instructions are centralized in
[`examples/env/README.md`](../../examples/env/README.md#vllm-metax-0230-sleep-mode).
This file documents only the implementation boundary so those operational
instructions have one authoritative source.

The replacement remains opt-in and version-specific. It is deliberately not
imported unconditionally by `verl-metax`: deployments that do not enable sleep
mode should continue using the installed vllm-metax package unchanged.
