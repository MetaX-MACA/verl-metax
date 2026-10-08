# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Version-bounded compatibility patches for exactly verl 0.9.0.

They are applied only by the ``verl-metax-ppo`` launcher, are idempotent, and
must be removed when equivalent public extension points exist in verl.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from verl_metax.platform import DEVICE_FLOPS
from verl_metax.sleep_mode_patch import PATCH_MODE_ENV

logger = logging.getLogger(__name__)

RAY_WORKER_SETUP_HOOK = "verl_metax.worker_setup.setup"


@dataclass(frozen=True)
class Workaround:
    identifier: str
    reason: str
    remove_when: str
    apply: Callable[[], bool]


def _mark_patched(function: Callable[..., Any]) -> Callable[..., Any]:
    function.__verl_metax_patched__ = True
    return function


def _is_patched(function: Callable[..., Any]) -> bool:
    return bool(getattr(function, "__verl_metax_patched__", False))


def patch_device_flops() -> bool:
    """Register C500/C550 theoretical BF16 FLOPS in verl's 0.9.0 table."""

    module = importlib.import_module("verl.utils.flops_counter")
    changed = any(module._DEVICE_FLOPS.get(name) != value for name, value in DEVICE_FLOPS.items())
    module._DEVICE_FLOPS.update(DEVICE_FLOPS)
    return changed


def patch_numa_affinity() -> bool:
    """Route verl's NVIDIA-only NUMA helper through PlatformMetaX."""

    module = importlib.import_module("verl.utils.distributed")
    original = module.set_numa_affinity
    if _is_patched(original):
        return False

    @_mark_patched
    @functools.wraps(original)
    def set_numa_affinity() -> None:
        from verl.plugin.platform.platform_manager import get_platform

        platform = get_platform()
        if platform.vendor_name != "metax":
            original()
            return

        try:
            import ray

            if ray.is_initialized():
                resource = platform.ray_resource_name()
                device_id = ray.get_runtime_context().get_accelerator_ids()[resource][0]
            else:
                device_id = os.environ["LOCAL_RANK"]
            platform.set_cpu_affinity(device_id)
        except Exception as exc:
            logger.warning("Failed to set MetaX NUMA affinity: %s", exc)

    module.set_numa_affinity = set_numa_affinity
    return True


def patch_trtllm_device_uuid() -> bool:
    """Make TensorRT-LLM resource mapping use the active platform UUID."""

    if importlib.util.find_spec("tensorrt_llm") is None:
        return False
    module = importlib.import_module("verl.workers.rollout.trtllm_rollout.trtllm_rollout")
    original = module.get_device_uuid
    if _is_patched(original):
        return False

    @_mark_patched
    @functools.wraps(original)
    def get_device_uuid(device_id: int | str) -> str:
        from verl.plugin.platform.platform_manager import get_platform

        return get_platform().get_device_uuid(device_id)

    module.get_device_uuid = get_device_uuid
    return True


def patch_vllm_run_headless() -> bool:
    """Avoid holding the stale function bound by verl before vLLM plugin discovery."""

    if importlib.util.find_spec("vllm") is None:
        return False
    module = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_async_server")
    original = getattr(module, "run_headless", None)
    if original is not None and _is_patched(original):
        return False

    @_mark_patched
    def run_headless(*args: Any, **kwargs: Any) -> Any:
        serve = importlib.import_module("vllm.entrypoints.cli.serve")
        return serve.run_headless(*args, **kwargs)

    module.run_headless = run_headless
    return True


def _shutdown_dataloader_workers(trainer: Any) -> None:
    """Best-effort shutdown for every persistent iterator owned by a trainer."""

    iterators = [getattr(trainer, "train_dataloader_it", None)]
    loaders = [
        getattr(trainer, "train_dataloader", None),
        getattr(trainer, "val_dataloader", None),
    ]
    iterators.extend(
        getattr(loader, "_iterator", None)
        for loader in loaders
        if loader is not None
    )

    seen: set[int] = set()
    for iterator in iterators:
        if iterator is None or id(iterator) in seen:
            continue
        seen.add(id(iterator))

        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                logger.warning("Failed to shut down DataLoader workers", exc_info=True)

    trainer.train_dataloader_it = None
    for loader in loaders:
        if loader is not None and hasattr(loader, "_iterator"):
            loader._iterator = None


def _ray_actor_source_class(actor_class: Any) -> type[Any] | None:
    """Return the user class hidden behind Ray's ``ActorClass`` wrapper."""

    metadata = getattr(actor_class, "__ray_metadata__", None)
    modified_class = getattr(metadata, "modified_class", None)
    return getattr(modified_class, "__ray_actor_class__", None)


def patch_dataloader_worker_cleanup() -> bool:
    """Release persistent DataLoader workers when the v1 task runner exits."""

    trainer_module = importlib.import_module("verl.trainer.ppo.v1.trainer_base")
    trainer_cls = trainer_module.PPOTrainer
    existing_close = getattr(trainer_cls, "close", None)

    # A native implementation takes precedence. This also avoids calling close
    # twice when running against a verl build that already carries c98d47b1.
    if existing_close is not None and not _is_patched(existing_close):
        return False

    changed = False
    if existing_close is None:

        @_mark_patched
        def close(self: Any) -> None:
            _shutdown_dataloader_workers(self)

        trainer_cls.close = close
        changed = True

    runner_module = importlib.import_module("verl.trainer.main_ppo")
    actor_class = runner_module.TaskRunnerV1
    runner_cls = _ray_actor_source_class(actor_class) or actor_class
    original_run = runner_cls.run
    if _is_patched(original_run):
        return changed

    @_mark_patched
    @functools.wraps(original_run)
    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original_run(self, *args, **kwargs)
        finally:
            trainer = getattr(self, "trainer", None)
            close = getattr(trainer, "close", None)
            if callable(close):
                close()
            elif trainer is not None:
                # Ray workers import verl independently, so the driver-side
                # PPOTrainer.close monkey patch is not present there.
                _shutdown_dataloader_workers(trainer)

    runner_cls.run = run

    if runner_cls is not actor_class:
        # TaskRunnerV1 is decorated with @ray.remote in verl 0.9.0. Rebuild
        # the ActorClass from its user class so Ray exports the patched run()
        # implementation to the worker process. Mutating ActorClass.run on the
        # driver only changes Ray's client-side method wrapper.
        ray = importlib.import_module("ray")
        runner_module.TaskRunnerV1 = ray.remote(runner_cls)

    return True


def patch_ray_worker_setup(selection: str) -> bool:
    """Inject the verl-metax bootstrap into Ray's final runtime environment."""

    ray = importlib.import_module("ray")
    original = ray.init
    if _is_patched(original):
        return False

    @_mark_patched
    @functools.wraps(original)
    def init(*args: Any, **kwargs: Any) -> Any:
        runtime_env = dict(kwargs.get("runtime_env") or {})
        existing_hook = runtime_env.get("worker_process_setup_hook")
        if existing_hook not in (None, RAY_WORKER_SETUP_HOOK):
            raise RuntimeError(
                "verl-metax requires Ray worker_process_setup_hook="
                f"{RAY_WORKER_SETUP_HOOK!r}, but another hook is configured: {existing_hook!r}"
            )

        env_vars = dict(runtime_env.get("env_vars") or {})
        env_vars["VERL_METAX_WORKAROUNDS"] = selection
        env_vars[PATCH_MODE_ENV] = os.getenv(PATCH_MODE_ENV, "auto")
        runtime_env["env_vars"] = env_vars
        runtime_env["worker_process_setup_hook"] = RAY_WORKER_SETUP_HOOK
        kwargs["runtime_env"] = runtime_env
        return original(*args, **kwargs)

    ray.init = init
    return True


def _uses_asm(config: Any) -> bool:
    try:
        actor = config.actor_rollout_ref.actor
        if actor.strategy != "megatron":
            return False
        overrides = actor.megatron.override_transformer_config
        return overrides.get("te_flash_backend", "cuda") == "asm"
    except (AttributeError, KeyError, TypeError):
        return False


@contextmanager
def asm_environment() -> Iterator[None]:
    """Enable ASM-only variables and restore the caller's environment."""

    values = {
        "MHA_USE_BLAS": "ON",
        "MHA_BWD_NO_ATOMIC_F64": "1",
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def patch_megatron_asm_scopes() -> bool:
    """Scope ASM variables to actor forward/backward operations."""

    changed = False
    targets = (
        ("verl.trainer.ppo.v1.trainer_base", "PPOTrainer"),
        ("verl.trainer.ppo.ray_trainer", "RayPPOTrainer"),
    )
    for module_name, class_name in targets:
        changed = patch_megatron_asm_module(module_name, class_name) or changed
    return changed


def patch_megatron_asm_module(module_name: str, class_name: str | None = None) -> bool:
    """Patch one already-imported trainer module without importing its peer."""

    if class_name is None:
        class_name = "PPOTrainer" if module_name.endswith(".v1.trainer_base") else "RayPPOTrainer"
    module = importlib.import_module(module_name)
    trainer_cls = getattr(module, class_name)
    changed = False
    for method_name in ("_compute_old_log_prob", "_update_actor"):
        original = getattr(trainer_cls, method_name)
        if _is_patched(original):
            continue

        def make_wrapper(method: Callable[..., Any]) -> Callable[..., Any]:
            @_mark_patched
            @functools.wraps(method)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                if not _uses_asm(self.config):
                    return method(self, *args, **kwargs)
                with asm_environment():
                    return method(self, *args, **kwargs)

            return wrapper

        setattr(trainer_cls, method_name, make_wrapper(original))
        changed = True
    return changed


def patch_entropy_kernel_tile() -> bool:
    """Apply the C500/C550-tested entropy-kernel tile before first compile."""

    if importlib.util.find_spec("triton") is None:
        return False
    triton = importlib.import_module("triton")
    module = importlib.import_module("verl.utils.kernel.kernels")
    kernel = module.efficient_entropy_kernel_general_mainloop
    configs = getattr(kernel, "configs", None)
    if not configs:
        logger.warning("Entropy kernel does not expose Triton autotune configs; tile workaround skipped")
        return False

    expected = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}
    if len(configs) == 1 and getattr(configs[0], "kwargs", None) == expected:
        return False

    kernel.configs = [triton.Config(expected, num_stages=3, num_warps=8)]
    cache = getattr(kernel, "cache", None)
    if hasattr(cache, "clear"):
        cache.clear()
    return True


WORKAROUNDS = (
    Workaround(
        "device-flops",
        "verl 0.9.0 has no platform hook for theoretical device FLOPS",
        "verl exposes a platform capability hook or carries C500/C550 values",
        patch_device_flops,
    ),
    Workaround(
        "numa-affinity",
        "verl 0.9.0 hard-codes NVML in its NUMA helper",
        "verl delegates NUMA affinity to PlatformBase",
        patch_numa_affinity,
    ),
    Workaround(
        "trtllm-device-uuid",
        "verl 0.9.0 hard-codes NVML for TensorRT-LLM device UUIDs",
        "the rollout delegates UUID lookup to PlatformBase",
        patch_trtllm_device_uuid,
    ),
    Workaround(
        "vllm-run-headless",
        "verl 0.9.0 binds run_headless before vLLM discovers the vllm-metax plugin",
        "verl resolves vllm.entrypoints.cli.serve.run_headless dynamically",
        patch_vllm_run_headless,
    ),
    Workaround(
        "dataloader-worker-cleanup",
        "verl 0.9.0 leaves persistent DataLoader workers alive when the v1 task runner exits",
        "verl provides and invokes a public trainer close lifecycle",
        patch_dataloader_worker_cleanup,
    ),
    Workaround(
        "megatron-asm-scopes",
        "ASM needs two environment variables only around actor forward/backward",
        "verl exposes actor operation hooks or ASM no longer needs the variables",
        patch_megatron_asm_scopes,
    ),
    Workaround(
        "entropy-kernel-tile",
        "the upstream entropy tile is not suitable for validated C500/C550 runs",
        "upstream autotuning selects the MetaX tile without a patch",
        patch_entropy_kernel_tile,
    ),
)

WORKER_WORKAROUND_IDS = {
    "device-flops",
    "numa-affinity",
    "trtllm-device-uuid",
    "vllm-run-headless",
    "megatron-asm-scopes",
    "entropy-kernel-tile",
}

WORKER_MODULE_WORKAROUNDS = {
    "verl.utils.flops_counter": "device-flops",
    "verl.utils.distributed": "numa-affinity",
    "verl.workers.rollout.trtllm_rollout.trtllm_rollout": "trtllm-device-uuid",
    "verl.workers.rollout.vllm_rollout.vllm_async_server": "vllm-run-headless",
    "verl.trainer.ppo.v1.trainer_base": "megatron-asm-scopes",
    "verl.trainer.ppo.ray_trainer": "megatron-asm-scopes",
    "verl.utils.kernel.kernels": "entropy-kernel-tile",
}


def _selected_identifiers(selection: str | None) -> tuple[str, set[str]]:
    if selection is None:
        selection = os.getenv("VERL_METAX_WORKAROUNDS", "all")
    selected = {item.strip() for item in selection.split(",") if item.strip()}
    return selection, selected


def _validate_selection(selected: set[str]) -> None:
    known = {workaround.identifier for workaround in WORKAROUNDS}
    unknown = selected - known - {"all", "default", "none", "off"}
    if unknown:
        raise ValueError(f"Unknown verl-metax workarounds: {sorted(unknown)}; available: {sorted(known)}")


def apply_worker_workarounds(selection: str | None = None) -> dict[str, bool]:
    """Apply process-local patches inside a Ray worker."""

    _, selected = _selected_identifiers(selection)
    _validate_selection(selected)
    if selected & {"none", "off"}:
        return {}

    results: dict[str, bool] = {}
    for workaround in WORKAROUNDS:
        if workaround.identifier not in WORKER_WORKAROUND_IDS:
            continue
        if not selected.intersection({"all", "default", workaround.identifier}):
            continue
        results[workaround.identifier] = workaround.apply()
    return results


def apply_imported_worker_workaround(module_name: str, selection: str | None = None) -> bool:
    """Apply only the workaround associated with a module after it imports."""

    identifier = WORKER_MODULE_WORKAROUNDS.get(module_name)
    if identifier is None:
        return False

    _, selected = _selected_identifiers(selection)
    _validate_selection(selected)
    if selected & {"none", "off"}:
        return False
    if not selected.intersection({"all", "default", identifier}):
        return False

    if identifier == "megatron-asm-scopes":
        return patch_megatron_asm_module(module_name)

    workaround = next(item for item in WORKAROUNDS if item.identifier == identifier)
    return workaround.apply()


def apply_workarounds(selection: str | None = None) -> dict[str, bool]:
    """Apply selected workarounds and return per-patch status."""

    selection, selected = _selected_identifiers(selection)
    _validate_selection(selected)
    if selected & {"none", "off"}:
        return {}

    patch_ray_worker_setup(selection)

    results: dict[str, bool] = {}
    for workaround in WORKAROUNDS:
        if not selected.intersection({"all", "default", workaround.identifier}):
            continue
        results[workaround.identifier] = workaround.apply()
    return results
