# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MetaX-aware entry point for verl 0.9.0 PPO training."""

from __future__ import annotations

import logging
import runpy
import sys

from verl_metax.compat import check_verl_compatibility

logger = logging.getLogger(__name__)


def prepare() -> dict[str, bool]:
    """Register MetaX and apply the exact-version compatibility layer."""
    check_verl_compatibility(strict=True)

    # Importing the plugin explicitly also supports launch environments that
    # disabled automatic entry-point discovery.
    from verl_metax.plugin import register
    from verl_metax.workarounds.verl_0_9 import apply_workarounds

    register()
    results = apply_workarounds()
    for identifier, applied in results.items():
        logger.info("workaround %s: %s", identifier, "applied" if applied else "skipped")
    return results


def main() -> None:
    """Apply bounded 0.9.0 compatibility patches, then enter verl's PPO CLI."""

    prepare()

    from verl.trainer.main_ppo import main as verl_main

    verl_main()


def run_module_main() -> None:
    """Run an arbitrary verl module after preparing MetaX compatibility."""

    if len(sys.argv) < 2:
        raise SystemExit("usage: verl-metax-run <python-module> [arguments ...]")
    module_name = sys.argv.pop(1)
    prepare()
    runpy.run_module(module_name, run_name="__main__")


if __name__ == "__main__":
    main()
