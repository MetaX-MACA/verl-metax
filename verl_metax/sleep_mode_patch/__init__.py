# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Version-gated vllm-metax sleep-mode compatibility package."""

from verl_metax.sleep_mode_patch._core import (
    PATCH_MODE_ENV,
    SUPPORTED_VLLM_METAX,
    SleepModePatchReport,
    activate,
    main,
    patch_module_paths,
)

__all__ = [
    "PATCH_MODE_ENV",
    "SUPPORTED_VLLM_METAX",
    "SleepModePatchReport",
    "activate",
    "main",
    "patch_module_paths",
]
