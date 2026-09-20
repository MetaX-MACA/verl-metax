# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Ray worker bootstrap for process-local verl compatibility patches."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def setup() -> None:
    """Install lazy patches without initializing an accelerator prematurely."""

    from verl_metax.deferred import install

    installed = install()
    logger.info("verl-metax lazy Ray worker bootstrap: %s", "installed" if installed else "already installed")
