# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Redirect affected vllm-metax imports to the validated sleep-mode fixes."""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
import warnings
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

PATCH_MODE_ENV = "VERL_METAX_SLEEP_PATCH"
SUPPORTED_VLLM_METAX = SpecifierSet("==0.23.0")

_PACKAGE_DIR = Path(__file__).resolve().parent
_PATCH_ROOT = _PACKAGE_DIR / "_vllm_metax_0_23_0"
_REDIRECTS = {
    "vllm_metax.platform": _PATCH_ROOT / "platform.py",
    "vllm_metax.device_allocator.cumem": _PATCH_ROOT / "device_allocator" / "cumem.py",
    "vllm_metax.patch.plugin_enhancement.device_allocator": (
        _PATCH_ROOT / "patch" / "plugin_enhancement" / "device_allocator.py"
    ),
}
_VALID_MODES = frozenset({"auto", "on", "off"})


@dataclass(frozen=True)
class SleepModePatchReport:
    mode: str
    installed_version: str | None
    active: bool
    reason: str


class _VllmMetaxPatchFinder(importlib.abc.MetaPathFinder):
    __verl_metax_sleep_patch__ = True

    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001
        patch_file = _REDIRECTS.get(fullname)
        if patch_file is None or not patch_file.is_file():
            return None
        return importlib.util.spec_from_file_location(fullname, patch_file)


def _finder_installed() -> bool:
    return any(getattr(finder, "__verl_metax_sleep_patch__", False) for finder in sys.meta_path)


def _installed_vllm_metax_version() -> str | None:
    try:
        return metadata.version("vllm-metax")
    except metadata.PackageNotFoundError:
        return None


def _normalize_mode(mode: str | None) -> str:
    selected = mode if mode is not None else os.getenv(PATCH_MODE_ENV, "auto")
    selected = selected.strip().lower()
    if selected not in _VALID_MODES:
        choices = ", ".join(sorted(_VALID_MODES))
        raise ValueError(f"{PATCH_MODE_ENV} must be one of: {choices}; found {selected!r}")
    return selected


def _is_supported(version: str) -> bool:
    try:
        return Version(version) in SUPPORTED_VLLM_METAX
    except InvalidVersion:
        return False


def _prepend_child_bootstrap_path() -> None:
    package_dir = str(_PACKAGE_DIR)
    parts = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    normalized = {os.path.normcase(os.path.abspath(part)) for part in parts}
    if os.path.normcase(os.path.abspath(package_dir)) not in normalized:
        os.environ["PYTHONPATH"] = os.pathsep.join([package_dir, *parts])


def patch_module_paths() -> dict[str, str]:
    """Return the import redirects included in this installed package."""

    return {module: str(path) for module, path in _REDIRECTS.items()}


def activate(mode: str | None = None) -> SleepModePatchReport:
    """Activate the patch in this process and propagate it to child processes."""

    selected = _normalize_mode(mode)
    installed = _installed_vllm_metax_version()

    if _finder_installed():
        return SleepModePatchReport(selected, installed, True, "already active")
    if selected == "off":
        return SleepModePatchReport(selected, installed, False, "disabled by configuration")
    if installed is None:
        message = "vllm-metax is not installed"
        if selected == "on":
            raise RuntimeError(message)
        return SleepModePatchReport(selected, installed, False, message)
    if not _is_supported(installed):
        message = f"vllm-metax {installed} is outside the supported range {SUPPORTED_VLLM_METAX}"
        if selected == "on":
            raise RuntimeError(message)
        return SleepModePatchReport(selected, installed, False, message)

    missing = [str(path) for path in _REDIRECTS.values() if not path.is_file()]
    if missing:
        message = "sleep-mode patch files are missing: " + ", ".join(missing)
        raise RuntimeError(message)

    imported = sorted(module for module in _REDIRECTS if module in sys.modules)
    if imported:
        message = "sleep-mode patch activated too late; modules already imported: " + ", ".join(imported)
        if selected == "on":
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return SleepModePatchReport(selected, installed, False, message)

    sys.meta_path.insert(0, _VllmMetaxPatchFinder())
    _prepend_child_bootstrap_path()
    return SleepModePatchReport(selected, installed, True, "activated")


def main() -> None:
    report = activate()
    print(f"mode: {report.mode}")
    print(f"vllm-metax: {report.installed_version or 'not installed'}")
    print(f"active: {report.active}")
    print(f"reason: {report.reason}")
    for module, path in patch_module_paths().items():
        print(f"{module}: {path}")
