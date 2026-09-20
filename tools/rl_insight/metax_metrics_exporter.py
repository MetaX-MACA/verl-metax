#!/usr/bin/env python3
"""MetaX mx-smi -> Prometheus exporter (standalone component).

Each cycle runs ONE `mx-smi --show-usage --show-memory --show-temperature
--show-board-power`, splits stdout into per-GPU blocks (GPU#N ...) and updates
six gauges labelled by gpu index:

    metax_gpu_utilization_percent         percent  Utilization -> GPU
    metax_gpu_memory_used_bytes           bytes    vis_vram used  (KB x 1024)
    metax_gpu_memory_total_bytes          bytes    vis_vram total (KB x 1024)
    metax_gpu_memory_utilization_percent  percent  vis_vram usage
    metax_gpu_temperature_celsius         celsius  Chip Temperature -> hotspot
    metax_gpu_power_watts                 watts    Board Power -> Total

Failure isolation: any per-cycle failure (timeout, non-zero exit, unparsable
block) only logs a warning and keeps the previously published values; the loop
continues. GPUs that disappear from mx-smi output have their stale series
removed instead of reporting frozen values forever.

Deployment: a standalone process started outside the trainer (systemd /
supervisor / container init), scraped by Prometheus as target <NODE_IP>:9101.
It deliberately does NOT collect CPU/host metrics (use the standard
node-exporter for those, see tools/rl_insight/README.md) and has no coupling to
the verl trainer lifecycle.

Parser validated against real mx-smi 2.2.9 output on an 8x MXC550 node.
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time

try:
    from prometheus_client import Gauge, start_http_server
except ImportError as exc:  # pragma: no cover - startup convenience
    sys.exit(f"prometheus_client is required (pip install prometheus-client): {exc}")

MXSMI_CANDIDATES = ("/opt/mxdriver/bin/mx-smi", "/usr/bin/mx-smi")
SMI_ARGS = ("--show-usage", "--show-memory", "--show-temperature", "--show-board-power")
SMI_TIMEOUT_S = 30

log = logging.getLogger("metax_metrics_exporter")

GPU_LABELS = ["gpu"]
G_UTIL = Gauge("metax_gpu_utilization_percent", "GPU core utilization percent", GPU_LABELS)
G_MEM_USED = Gauge("metax_gpu_memory_used_bytes", "vis_vram used bytes", GPU_LABELS)
G_MEM_TOTAL = Gauge("metax_gpu_memory_total_bytes", "vis_vram total bytes", GPU_LABELS)
G_MEM_UTIL = Gauge("metax_gpu_memory_utilization_percent", "vis_vram usage percent", GPU_LABELS)
G_TEMP = Gauge("metax_gpu_temperature_celsius", "chip hotspot temperature celsius", GPU_LABELS)
G_POWER = Gauge("metax_gpu_power_watts", "board total power watts", GPU_LABELS)

ALL_GAUGES = (G_UTIL, G_MEM_USED, G_MEM_TOTAL, G_MEM_UTIL, G_TEMP, G_POWER)


def prune_absent_gpus(published: set[int], current: set[int]) -> None:
    """Remove series of GPUs that vanished from mx-smi output (stale values)."""
    for gone in sorted(published - current):
        for gauge in ALL_GAUGES:
            try:
                gauge.remove(str(gone))
            except (KeyError, ValueError):  # child already absent
                pass
        log.info("gpu %d no longer reported by mx-smi, removed stale series", gone)


# (gauge, compiled pattern searched within one GPU block, unit multiplier).
# Anchored on mx-smi section headers; validated against mx-smi 2.2.9 (MXC550).
# NOTE: Board Power lists "Way 0" BEFORE "Total", so a naive
# "Board Power\n Total" anchor never matches; use a lazy DOTALL match instead.
FIELD_PATTERNS = (
    (G_UTIL, re.compile(r"\bUtilization\s*\n\s+GPU\s+:\s*([\d.]+)\s*%"), 1.0),
    (G_MEM_USED, re.compile(r"\bvis_vram\s+used\s*:\s*(\d+)\s*KB"), 1024.0),
    (G_MEM_TOTAL, re.compile(r"\bvis_vram\s+total\s*:\s*(\d+)\s*KB"), 1024.0),
    (G_MEM_UTIL, re.compile(r"\bvis_vram\s+usage\s*:\s*([\d.]+)\s*%"), 1.0),
    (
        G_TEMP,
        re.compile(r"Chip Temperature\s*\n\s+hotspot[^:\n]*\s*:\s*([\d.]+)"),
        1.0,
    ),
    (
        G_POWER,
        re.compile(r"\bBoard Power\b.*?\bTotal\s*:\s*([\d.]+)\s*W", re.DOTALL),
        1.0,
    ),
)

RE_GPU_HEADER = re.compile(r"^GPU#(\d+)\b.*$", re.MULTILINE)


def find_mxsmi() -> str:
    found = shutil.which("mx-smi")
    if found:
        return found
    for cand in MXSMI_CANDIDATES:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError("mx-smi not found in PATH nor standard locations")


def split_gpu_blocks(text: str) -> dict[int, str]:
    """Map gpu index -> body text of its block (text after the header line)."""
    headers = list(RE_GPU_HEADER.finditer(text))
    blocks: dict[int, str] = {}
    for i, match in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        blocks[int(match.group(1))] = text[match.end():end]
    return blocks


def publish_block(gpu_index: int, body: str) -> int:
    updated = 0
    for gauge, pattern, multiplier in FIELD_PATTERNS:
        match = pattern.search(body)
        if match is None:
            log.warning("gpu %d: field pattern not matched: %s", gpu_index, pattern)
            continue
        gauge.labels(gpu=str(gpu_index)).set(float(match.group(1)) * multiplier)
        updated += 1
    return updated


def sample_once(mxsmi_path: str) -> tuple[set[int], int]:
    """Run mx-smi once, update gauges. Returns (gpu_indices_seen, fields_set)."""
    proc = subprocess.run(
        [mxsmi_path, *SMI_ARGS],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=SMI_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"mx-smi exited {proc.returncode}: {proc.stderr.strip()[:200]}")
    blocks = split_gpu_blocks(proc.stdout)
    if not blocks:
        raise RuntimeError("mx-smi produced no GPU# blocks")
    seen: set[int] = set()
    fields_set = 0
    for gpu_index, body in blocks.items():
        seen.add(gpu_index)
        try:
            fields_set += publish_block(gpu_index, body)
        except Exception:  # never let one bad block kill the loop
            log.exception("gpu %d: failed to parse block, skipping", gpu_index)
    return seen, fields_set


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MetaX mx-smi Prometheus exporter (standalone component)"
    )
    parser.add_argument("--port", type=int, default=9101, help="listen port (default 9101)")
    parser.add_argument("--interval", type=float, default=5.0, help="sampling interval seconds (default 5)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    mxsmi_path = find_mxsmi()
    log.info("using mx-smi at %s", mxsmi_path)

    start_http_server(args.port, addr="0.0.0.0")  # noqa: S104 - bind all interfaces by design
    log.info("serving metrics on 0.0.0.0:%d, sampling every %.1fs", args.port, args.interval)

    published: set[int] = set()
    try:
        while True:
            try:
                seen, fields_set = sample_once(mxsmi_path)
            except Exception as exc:
                # timeout / non-zero exit / parse errors: warn, keep old values
                log.warning("sample failed, keeping last published values: %s", exc)
            else:
                prune_absent_gpus(published, seen)
                published |= seen
                # per-cycle OK chatter goes to DEBUG to avoid flooding logs on
                # long-running deployments; failures stay at WARNING.
                log.debug("sample ok: %d gpus, %d fields updated", len(seen), fields_set)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("stopped by signal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
