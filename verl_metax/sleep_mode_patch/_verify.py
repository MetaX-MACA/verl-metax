# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Verify sleep-mode redirects in the driver and a real Ray worker."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from verl_metax.sleep_mode_patch import activate, patch_module_paths


def _inspect_process() -> dict[str, Any]:
    report = activate()
    expected = {name: str(Path(path).resolve()) for name, path in patch_module_paths().items()}
    imported: dict[str, str] = {}

    if report.active:
        for name in expected:
            module = importlib.import_module(name)
            imported[name] = str(Path(module.__file__).resolve())

    return {
        "mode": report.mode,
        "vllm-metax": report.installed_version,
        "active": report.active,
        "reason": report.reason,
        "expected": expected,
        "imported": imported,
        "matches": report.active and imported == expected,
    }


def _check(location: str, result: dict[str, Any]) -> None:
    if not result["active"]:
        raise SystemExit(f"FAIL: sleep-mode patch is inactive in {location}: {result['reason']}")
    if not result["matches"]:
        raise SystemExit(f"FAIL: {location} imported files do not match the packaged redirects")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="auto", help="Ray address passed to ray.init")
    parser.add_argument("--num-gpus", type=float, default=0, help="GPU resources reserved by the probe worker")
    parser.add_argument("--skip-ray", action="store_true", help="Only inspect the current Python process")
    parser.add_argument("--all-nodes", action="store_true", help="Run one probe on every alive Ray node")
    args = parser.parse_args()

    # prepare() installs the normal verl 0.9 workarounds and the Ray worker
    # setup hook before ray.init() creates the job runtime environment.
    from verl_metax.launcher import prepare

    prepare()
    driver = _inspect_process()
    print("driver:")
    print(json.dumps(driver, indent=2, sort_keys=True))
    _check("driver", driver)

    if args.skip_ray:
        print("PASS: the driver imports all three packaged sleep-mode replacements")
        return

    import ray

    ray.init(address=args.address)
    try:
        inspect_worker = ray.remote(num_gpus=args.num_gpus)(_inspect_process)
        if args.all_nodes:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            nodes = [node for node in ray.nodes() if node["Alive"]]
            if not nodes:
                raise SystemExit("FAIL: the connected Ray cluster has no alive nodes")
            refs = [
                inspect_worker.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
                ).remote()
                for node in nodes
            ]
            workers = list(zip(nodes, ray.get(refs), strict=True))
        else:
            workers = [({"NodeID": "scheduled", "NodeManagerAddress": "any"}, ray.get(inspect_worker.remote()))]
    finally:
        ray.shutdown()

    for node, worker in workers:
        location = f"Ray worker {node['NodeManagerAddress']} ({node['NodeID']})"
        print(f"{location}:")
        print(json.dumps(worker, indent=2, sort_keys=True))
        _check(location, worker)
    print("PASS: the driver and checked Ray workers import all three packaged sleep-mode replacements")


if __name__ == "__main__":
    main()
