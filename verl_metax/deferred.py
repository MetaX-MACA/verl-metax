# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Lazy import hooks for Ray worker-local compatibility patches.

This module intentionally imports only the Python standard library. Ray runs
its worker setup hook before executing an actor creation task, so importing
torch/verl/vLLM there can initialize an accelerator before actor-specific
device visibility has been installed.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any

TARGET_MODULES = {
    "verl.utils.flops_counter",
    "verl.utils.distributed",
    "verl.workers.rollout.trtllm_rollout.trtllm_rollout",
    "verl.workers.rollout.vllm_rollout.vllm_async_server",
    "verl.trainer.ppo.v1.trainer_base",
    "verl.trainer.ppo.ray_trainer",
    "verl.utils.kernel.kernels",
}


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, wrapped: importlib.abc.Loader) -> None:
        self.fullname = fullname
        self.wrapped = wrapped

    def create_module(self, spec: Any) -> ModuleType | None:
        create_module = getattr(self.wrapped, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)
        from verl_metax.workarounds.verl_0_9 import apply_imported_worker_workaround

        apply_imported_worker_workaround(self.fullname)


class _PatchFinder(importlib.abc.MetaPathFinder):
    __verl_metax_worker_finder__ = True

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in TARGET_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None or isinstance(spec.loader, _PatchLoader):
            return spec
        if not hasattr(spec.loader, "exec_module"):
            return spec
        spec.loader = _PatchLoader(fullname, spec.loader)
        return spec


def install() -> bool:
    """Install the lazy worker patch finder once without importing verl/torch."""

    if any(getattr(finder, "__verl_metax_worker_finder__", False) for finder in sys.meta_path):
        return False
    sys.meta_path.insert(0, _PatchFinder())
    return True
