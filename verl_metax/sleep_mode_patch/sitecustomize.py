# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Activate the packaged sleep-mode fix in spawned Python interpreters."""

try:
    from verl_metax.sleep_mode_patch import activate

    activate()
except Exception as exc:  # Python startup must remain usable for diagnostics.
    import warnings

    warnings.warn(f"verl-metax sleep-mode patch bootstrap failed: {exc}", RuntimeWarning, stacklevel=1)
