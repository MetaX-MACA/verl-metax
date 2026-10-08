# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Entry point loaded by verl through the ``verl.plugins`` group."""

from __future__ import annotations

import logging
import warnings

from verl.plugin.platform.platform_manager import PlatformRegistry

from verl_metax.compat import check_verl_compatibility
from verl_metax.platform import PlatformMetaX
from verl_metax.sleep_mode_patch import activate as activate_sleep_mode_patch

logger = logging.getLogger(__name__)
_registered = False


def register() -> type[PlatformMetaX]:
    """Register MetaX once and return the platform class for diagnostics."""

    global _registered
    if _registered:
        return PlatformMetaX

    # The plugin is deliberately version-bounded.  Failing here prevents a
    # partially compatible platform from being registered under another verl
    # release when entry-point auto-discovery is enabled.
    check_verl_compatibility(strict=True)
    sleep_patch = activate_sleep_mode_patch()
    logger.info("vllm-metax sleep-mode patch: %s", sleep_patch.reason)

    existing = PlatformRegistry.get("metax")
    if existing is not None and existing is not PlatformMetaX:
        warnings.warn(
            "Another plugin already registered the 'metax' platform. "
            "Use VERL_USE_EXTERNAL_PLUGINS=metax to avoid ambiguous plugin ordering.",
            RuntimeWarning,
            stacklevel=2,
        )

    PlatformRegistry.register(platform="metax")(PlatformMetaX)
    _registered = True
    logger.info("Registered verl-metax platform")
    return PlatformMetaX


# importlib.metadata.EntryPoint.load() imports the target but does not call it.
register()
