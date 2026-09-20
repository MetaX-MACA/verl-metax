# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Small, lazy wrapper around pymxsml.

Keeping pymxsml behind this module prevents importing the vendor management
library in processes that never query a UUID or set CPU affinity.
"""

from __future__ import annotations

import importlib
import threading
from types import ModuleType

_lock = threading.Lock()
_module: ModuleType | None = None
_initialized = False


class MXSMLUnavailable(RuntimeError):
    """Raised when the MetaX management library cannot be used."""


def _load() -> ModuleType:
    global _module
    if _module is None:
        try:
            _module = importlib.import_module("pymxsml")
        except ImportError as exc:
            raise MXSMLUnavailable(
                "pymxsml is required for MetaX UUID and NUMA operations; install it from the MACA runtime."
            ) from exc
    return _module


def initialize() -> ModuleType:
    """Initialize MXSML once per process and return its Python module."""

    global _initialized
    module = _load()
    with _lock:
        if not _initialized:
            try:
                flags = getattr(module, "MXSML_INIT_FLAG_REINIT", 0)
                module.mxSmlInitWithFlags(flags)
            except Exception as exc:
                raise MXSMLUnavailable(f"Failed to initialize pymxsml: {exc}") from exc
            _initialized = True
    return module


def get_device_uuid(device_id: int | str) -> str:
    """Return the MXSML UUID for a global device index."""

    module = initialize()
    try:
        info = module.mxSmlGetDeviceInfo(int(device_id))
        value = info.uuid
    except Exception as exc:
        raise MXSMLUnavailable(f"Failed to get the UUID for MetaX device {device_id}: {exc}") from exc

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise MXSMLUnavailable(f"Unexpected UUID type for MetaX device {device_id}: {type(value).__name__}")


def set_cpu_affinity(device_id: int | str) -> None:
    """Bind the current process to CPUs local to a MetaX device."""

    module = initialize()
    try:
        module.mxSmlSetCpuAffinity(int(device_id))
    except Exception as exc:
        raise MXSMLUnavailable(f"Failed to set CPU affinity for MetaX device {device_id}: {exc}") from exc
