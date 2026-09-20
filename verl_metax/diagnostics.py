# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Command-line diagnostics that avoid initializing accelerator devices."""

from __future__ import annotations

import importlib.util
import shutil
from importlib import metadata

from verl_metax import __version__
from verl_metax.compat import check_verl_compatibility


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def collect() -> dict[str, str]:
    report = check_verl_compatibility(strict=False)
    return {
        "verl-metax": __version__,
        "verl": report.installed or "not installed",
        "verl supported": report.supported,
        "verl compatible": "unknown" if report.compatible is None else str(report.compatible),
        "numpy": _distribution_version("numpy"),
        "torch": _distribution_version("torch"),
        "ray": _distribution_version("ray"),
        "transformers": _distribution_version("transformers"),
        "megatron-core": _distribution_version("megatron-core"),
        "vllm-metax": _distribution_version("vllm-metax"),
        "pymxsml": "available" if importlib.util.find_spec("pymxsml") else "not available",
        "mx-smi": shutil.which("mx-smi") or "not found",
    }


def main() -> None:
    for key, value in collect().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
