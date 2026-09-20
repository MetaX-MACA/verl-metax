# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility checks kept separate from plugin registration."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from importlib import metadata

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

SUPPORTED_VERL = SpecifierSet("==0.9.0")


@dataclass(frozen=True)
class CompatibilityReport:
    component: str
    installed: str | None
    supported: str
    compatible: bool | None


def _read_verl_version() -> str | None:
    try:
        return metadata.version("verl")
    except metadata.PackageNotFoundError:
        try:
            import verl

            return getattr(verl, "__version__", None)
        except ImportError:
            return None


def check_verl_compatibility(*, strict: bool | None = None) -> CompatibilityReport:
    """Validate the installed verl version and optionally fail closed."""

    installed = _read_verl_version()
    compatible: bool | None
    if installed is None:
        compatible = None
    else:
        try:
            compatible = Version(installed) in SUPPORTED_VERL
        except InvalidVersion:
            compatible = None

    report = CompatibilityReport(
        component="verl",
        installed=installed,
        supported=str(SUPPORTED_VERL),
        compatible=compatible,
    )

    if strict is None:
        strict = os.getenv("VERL_METAX_STRICT_COMPAT", "0").strip().lower() in {"1", "true", "yes", "on"}

    if compatible is False:
        message = f"verl-metax supports verl {SUPPORTED_VERL}; found {installed}."
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    return report
