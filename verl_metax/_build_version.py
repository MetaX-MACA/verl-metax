# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Build-time package version derived from the validated verl release."""

from __future__ import annotations

import os
import re

BASE_VERSION = "0.9.0"
_LOCAL_VERSION_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")


def package_version(maca_ai_version: str | None = None) -> str:
    """Return a PEP 440 version with an optional MACA AI local identifier."""

    raw_version = maca_ai_version
    if raw_version is None:
        raw_version = os.getenv("MACA_AI_VERSION")
    if raw_version is None or not raw_version.strip():
        return BASE_VERSION

    raw_version = raw_version.strip()
    if _LOCAL_VERSION_PATTERN.fullmatch(raw_version) is None:
        raise ValueError(
            "MACA_AI_VERSION must contain alphanumeric components separated "
            "by '.', '-' or '_'"
        )

    normalized_version = re.sub(r"[._-]+", ".", raw_version).lower()
    return f"{BASE_VERSION}+maca{normalized_version}"


BUILD_VERSION = package_version()
