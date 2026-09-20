# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Standalone mx-smi to Prometheus exporter."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from typing import Any

from verl_metax.observability.mxsmi import GPUMetrics, parse_mxsmi_output

MXSMI_CANDIDATES = ("/opt/mxdriver/bin/mx-smi", "/usr/bin/mx-smi")
SMI_ARGS = ("--show-usage", "--show-memory", "--show-temperature", "--show-board-power")
SMI_TIMEOUT_SECONDS = 30

logger = logging.getLogger("verl_metax.metrics")


def find_mxsmi() -> str:
    found = shutil.which("mx-smi")
    if found:
        return found
    for candidate in MXSMI_CANDIDATES:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError("mx-smi was not found in PATH or a standard driver location")


def sample(mxsmi_path: str, *, timeout: float = SMI_TIMEOUT_SECONDS) -> dict[int, GPUMetrics]:
    process = subprocess.run(
        [mxsmi_path, *SMI_ARGS],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"mx-smi exited {process.returncode}: {process.stderr.strip()[:200]}")
    metrics = parse_mxsmi_output(process.stdout)
    if not metrics:
        raise RuntimeError("mx-smi produced no GPU# blocks")
    return metrics


class PrometheusPublisher:
    """Own Prometheus collectors so importing the parser has no side effects."""

    FIELDS = {
        "utilization_percent": ("metax_gpu_utilization_percent", "GPU core utilization percent"),
        "memory_used_bytes": ("metax_gpu_memory_used_bytes", "Visible VRAM used bytes"),
        "memory_total_bytes": ("metax_gpu_memory_total_bytes", "Visible VRAM total bytes"),
        "memory_utilization_percent": (
            "metax_gpu_memory_utilization_percent",
            "Visible VRAM utilization percent",
        ),
        "temperature_celsius": ("metax_gpu_temperature_celsius", "Chip hotspot temperature Celsius"),
        "power_watts": ("metax_gpu_power_watts", "Board total power watts"),
    }

    def __init__(self, *, registry: Any = None):
        try:
            from prometheus_client import Gauge
        except ImportError as exc:
            raise RuntimeError("Install verl-metax[observability] to run the metrics exporter") from exc

        kwargs = {} if registry is None else {"registry": registry}
        self.gauges = {
            field: Gauge(name, description, ["gpu"], **kwargs)
            for field, (name, description) in self.FIELDS.items()
        }
        self.published: set[int] = set()

    def publish(self, metrics: dict[int, GPUMetrics]) -> None:
        current = set(metrics)
        for missing in sorted(self.published - current):
            for gauge in self.gauges.values():
                try:
                    gauge.remove(str(missing))
                except (KeyError, ValueError):
                    pass

        for index, sample_values in metrics.items():
            for field, gauge in self.gauges.items():
                value = getattr(sample_values, field)
                if value is not None:
                    gauge.labels(gpu=str(index)).set(value)
        self.published = current


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export mx-smi metrics for Prometheus")
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--address", default="0.0.0.0")
    arguments = parser.parse_args(argv)

    try:
        from prometheus_client import start_http_server
    except ImportError as exc:
        parser.error(f"Install verl-metax[observability]: {exc}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    mxsmi_path = find_mxsmi()
    publisher = PrometheusPublisher()
    start_http_server(arguments.port, addr=arguments.address)
    logger.info("serving on %s:%d using %s", arguments.address, arguments.port, mxsmi_path)

    try:
        while True:
            try:
                publisher.publish(sample(mxsmi_path))
            except Exception as exc:
                logger.warning("sample failed; keeping the previous metric values: %s", exc)
            time.sleep(arguments.interval)
    except KeyboardInterrupt:
        logger.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
