# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Parse metrics emitted by ``mx-smi`` 2.2.9."""

from __future__ import annotations

import re
from dataclasses import dataclass

GPU_HEADER = re.compile(r"^GPU#(\d+)\b.*$", re.MULTILINE)


@dataclass(frozen=True)
class GPUMetrics:
    index: int
    utilization_percent: float | None = None
    memory_used_bytes: float | None = None
    memory_total_bytes: float | None = None
    memory_utilization_percent: float | None = None
    temperature_celsius: float | None = None
    power_watts: float | None = None


FIELD_PATTERNS: dict[str, tuple[re.Pattern[str], float]] = {
    "utilization_percent": (re.compile(r"\bUtilization\s*\n\s+GPU\s+:\s*([\d.]+)\s*%"), 1.0),
    "memory_used_bytes": (re.compile(r"\bvis_vram\s+used\s*:\s*(\d+)\s*KB"), 1024.0),
    "memory_total_bytes": (re.compile(r"\bvis_vram\s+total\s*:\s*(\d+)\s*KB"), 1024.0),
    "memory_utilization_percent": (re.compile(r"\bvis_vram\s+usage\s*:\s*([\d.]+)\s*%"), 1.0),
    "temperature_celsius": (
        re.compile(r"Chip Temperature\s*\n\s+hotspot[^:\n]*\s*:\s*([\d.]+)"),
        1.0,
    ),
    "power_watts": (re.compile(r"\bBoard Power\b.*?\bTotal\s*:\s*([\d.]+)\s*W", re.DOTALL), 1.0),
}


def split_gpu_blocks(text: str) -> dict[int, str]:
    """Return the body of each ``GPU#N`` section."""

    headers = list(GPU_HEADER.finditer(text))
    blocks: dict[int, str] = {}
    for position, match in enumerate(headers):
        end = headers[position + 1].start() if position + 1 < len(headers) else len(text)
        blocks[int(match.group(1))] = text[match.end() : end]
    return blocks


def parse_mxsmi_output(text: str) -> dict[int, GPUMetrics]:
    """Parse all recognized fields while tolerating fields absent in a block."""

    parsed: dict[int, GPUMetrics] = {}
    for index, body in split_gpu_blocks(text).items():
        values: dict[str, float | int | None] = {"index": index}
        for field, (pattern, multiplier) in FIELD_PATTERNS.items():
            match = pattern.search(body)
            values[field] = None if match is None else float(match.group(1)) * multiplier
        parsed[index] = GPUMetrics(**values)
    return parsed
