# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MetaX hardware integration for verl."""

from importlib import metadata

from verl_metax._build_version import BASE_VERSION

try:
    __version__ = metadata.version("verl-metax")
except metadata.PackageNotFoundError:
    # Source checkouts have no distribution metadata until they are installed.
    __version__ = BASE_VERSION

__all__ = ["__version__"]
