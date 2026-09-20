# Workaround policy

A workaround is accepted only when the public verl extension interfaces cannot
express the required behavior. Every workaround must document:

- the observed failure and a reproducer;
- affected verl, MACA, torch, and backend versions;
- why a platform or engine extension cannot solve it;
- an upstream issue or pull request when the problem is generic;
- the condition and target release for removal;
- a focused regression test.

verl workarounds are isolated under `verl_metax.workarounds`; they must not
overwrite the installed `verl` package on disk. Import-time monkey patches are
a last resort and must be version-gated.

The supported compatibility mechanisms are:

| Change | Destination |
| --- | --- |
| MXSML UUID and NUMA implementation | MetaX platform/integration |
| Generic UUID/NUMA call sites | upstream verl |
| Stale vLLM `run_headless` binding | versioned workaround until verl resolves it dynamically |
| vLLM 0.23 sleep-mode defects | versioned full-file replacement loaded before Python starts |
| ASM environment toggles | versioned workaround until an engine hook exists |
| Triton tile-size change | versioned, benchmark-backed workaround |
| DataLoader worker cleanup | versioned workaround until verl exposes and invokes a trainer close lifecycle |
| Hydra package search paths | upstream verl |

The sleep-mode replacement is not a verl monkey patch. The helper at
`examples/env/vllm-metax-0.23-sleep-mode.sh` prepends the versioned patch
directory to `PYTHONPATH`; its `sitecustomize.py` then replaces exactly three
`vllm_metax` Python modules while leaving the installed package intact. See
the [shared environment guide](../examples/env/README.md#vllm-metax-0230-sleep-mode)
for activation and verification, and
[`patches/vllm_metax_0_23_0/README.md`](../patches/vllm_metax_0_23_0/README.md)
for implementation ownership.

## DataLoader worker cleanup

verl 0.9.0 can leave persistent train or validation DataLoader workers alive
until the Ray trainer actor is torn down, producing messages such as
`DataLoader worker ... is killed by signal: Killed`. The
`dataloader-worker-cleanup` workaround adds an idempotent `PPOTrainer.close()`
and invokes the same cleanup from `TaskRunnerV1.run()` on both normal and
exceptional exits. Because `TaskRunnerV1` is already decorated with
`@ray.remote` when imported, the workaround patches its underlying user class
and rebuilds the Ray actor class; changing the driver-side method proxy alone
does not modify the code executed by the Ray worker. The remote cleanup does
not depend on the driver-side `PPOTrainer.close()` monkey patch. It deduplicates
shared iterators, tolerates workers that have already exited, and clears the
strong `_iterator` references held by StatefulDataLoader.

This workaround necessarily uses PyTorch's private `_shutdown_workers` and
`_iterator` attributes because verl 0.9.0 exposes no trainer close hook. Remove
it when the supported verl release provides and invokes an equivalent public
lifecycle. If a native `PPOTrainer.close()` is detected, verl-metax leaves it
untouched and does not wrap the task runner.

`VERL_METAX_WORKAROUNDS=all` (the default) applies the verl 0.9.0 compatibility
set. `default` is an alias for `all`, a comma-separated identifier list selects
individual workarounds, and `none` disables every verl workaround. This setting
does not control the separate vllm-metax sleep-mode replacement.

## Ray worker propagation

Most verl training code executes in fresh Ray worker processes. Patching a
module only in the launcher process does not change a module imported by those
workers. The launcher therefore injects `verl_metax.worker_setup.setup` through
Ray's `worker_process_setup_hook`. The setup hook installs a standard-library-
only import hook; it deliberately does not import verl, torch, vLLM, or Triton,
because doing so before Ray applies actor-specific device visibility can bind
multiple ranks to the same physical accelerator. Each process-local workaround
is applied lazily after its target module is imported in the assigned actor.
The selected `VERL_METAX_WORKAROUNDS` value is forwarded in the job runtime
environment.

An explicitly configured third-party `worker_process_setup_hook` currently
conflicts with this bootstrap and produces an error instead of silently losing
either hook. Run the real-worker verification after changing the launcher,
workaround selection, Ray version, or container image:

```bash
python tools/verify_workarounds.py --address auto --num-gpus 1
```
