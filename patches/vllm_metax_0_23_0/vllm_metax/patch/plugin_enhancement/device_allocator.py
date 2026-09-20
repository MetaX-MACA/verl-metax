# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Replace CUDA `CuMemAllocator` with the MetaX allocator to support
#       sleep mode on MACA.
#
# Affected versions: v0.21.0
# -----------------------------------------------

from vllm.device_allocator import MemAllocator


def get_mem_allocator_instance() -> MemAllocator:
    from vllm_metax.device_allocator.cumem import CuMemAllocator

    return CuMemAllocator.get_instance()


import sys

import vllm.device_allocator

try:
    # Make vllm core resolve the plugin's cumem module (whose .so is
    # MACA-compatible) instead of vllm's own, which cannot load on MACA.
    import vllm_metax.device_allocator.cumem as _mx_cumem

    sys.modules["vllm.device_allocator.cumem"] = _mx_cumem
except ImportError:
    pass

vllm.device_allocator.get_mem_allocator_instance = get_mem_allocator_instance

