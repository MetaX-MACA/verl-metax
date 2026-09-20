# Copyright (c) 2026 BAAI. All rights reserved.
# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from the MetaX platform reference implementation in
# verl-project/verl-hardware-plugin:
# https://github.com/verl-project/verl-hardware-plugin/blob/main/verl_hardware_plugin/platforms/platform_cuda_metax.py

"""MetaX implementation of verl's hardware platform contract."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any

import torch
from verl.plugin.platform.platform_base import PlatformBase

from verl_metax.integrations import mxsml

logger = logging.getLogger(__name__)

DEVICE_FLOPS = {
    "C500": 240e12,
    "C550": 280e12,
}


class PlatformMetaX(PlatformBase):
    """CUDA-compatible MetaX platform backed by MACA and MCCL."""

    @property
    def device_name(self) -> str:
        return "cuda"

    @property
    def vendor_name(self) -> str:
        return "metax"

    @property
    def device_module(self) -> ModuleType:
        return torch.cuda

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def is_platform_available(self, use_smi_check: bool = False) -> bool:
        # SMI probing also works in CPU-only Ray actors where CUDA devices have
        # deliberately been hidden from the process.
        if use_smi_check:
            return self.check_smi_command("mx-smi")
        return self.is_available()

    def current_device(self) -> int:
        return torch.cuda.current_device()

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def set_device(self, device_index: int) -> None:
        torch.cuda.set_device(device_index)

    def synchronize(self, device_index: int | None = None) -> None:
        if device_index is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device_index)

    def manual_seed(self, seed: int) -> None:
        torch.cuda.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.cuda.manual_seed_all(seed)

    def set_allocator_settings(self, settings: str) -> None:
        try:
            torch.cuda.memory._set_allocator_settings(settings)
        except (AttributeError, RuntimeError) as exc:
            logger.warning("MetaX allocator settings were not applied: %s", exc)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def get_device_capability(self, device_index: int = 0) -> tuple[int | None, int | None]:
        return torch.cuda.get_device_capability(device_index)

    def communication_backend_name(self) -> str:
        # torch+metax exposes MCCL through PyTorch's NCCL-compatible backend.
        return "nccl"

    def visible_devices_envvar(self) -> str:
        return "CUDA_VISIBLE_DEVICES"

    def get_device_uuid(self, device_id: int | str) -> str:
        return mxsml.get_device_uuid(device_id)

    def set_cpu_affinity(self, device_id: int | str) -> None:
        """MetaX extension point proposed for adoption by verl core."""

        mxsml.set_cpu_affinity(device_id)

    def ray_resource_name(self) -> str:
        return "GPU"

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"num_gpus": num_gpus}

    def is_ipc_supported(self) -> bool:
        return True

    @contextmanager
    def nvtx_range(self, msg: str) -> Iterator[None]:
        torch.cuda.nvtx.range_push(msg)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def profiler_start(self) -> None:
        torch.cuda.cudart().cudaProfilerStart()

    def profiler_stop(self) -> None:
        torch.cuda.cudart().cudaProfilerStop()

    def get_collective_module(self) -> Any:
        try:
            from cupy.cuda import nccl
        except ImportError:
            return None
        return nccl

    def cudart(self) -> Any:
        return torch.cuda.cudart()
