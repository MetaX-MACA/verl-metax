#!/usr/bin/env python3
# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Verify that verl-metax compatibility patches reach a real Ray worker."""

from __future__ import annotations

import argparse
import importlib.util
import json
from typing import Any

import ray

from verl_metax.launcher import prepare


EXPECTED_TILE = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}
NOT_INSTALLED = "not-installed"


def _is_patched(value: Any) -> bool:
    return bool(getattr(value, "__verl_metax_patched__", False))


def _inspect_worker() -> dict[str, Any]:
    import verl.utils.distributed as distributed
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    from verl.utils.flops_counter import _DEVICE_FLOPS
    from verl.utils.kernel import kernels

    configs = kernels.efficient_entropy_kernel_general_mainloop.configs
    result: dict[str, Any] = {
        "numa-affinity": _is_patched(distributed.set_numa_affinity),
        "device-flops": {
            "C500": _DEVICE_FLOPS.get("C500"),
            "C550": _DEVICE_FLOPS.get("C550"),
        },
        "entropy-kernel-tile": [getattr(config, "kwargs", None) for config in configs],
        "megatron-asm-scopes-v1": {
            "_compute_old_log_prob": _is_patched(PPOTrainer._compute_old_log_prob),
            "_update_actor": _is_patched(PPOTrainer._update_actor),
        },
        "megatron-asm-scopes-v0": {
            "_compute_old_log_prob": _is_patched(RayPPOTrainer._compute_old_log_prob),
            "_update_actor": _is_patched(RayPPOTrainer._update_actor),
        },
    }

    if importlib.util.find_spec("vllm") is None:
        result["vllm-run-headless"] = NOT_INSTALLED
    else:
        from verl.workers.rollout.vllm_rollout import vllm_async_server

        result["vllm-run-headless"] = _is_patched(vllm_async_server.run_headless)

    if importlib.util.find_spec("tensorrt_llm") is None:
        result["trtllm-device-uuid"] = NOT_INSTALLED
    else:
        from verl.workers.rollout.trtllm_rollout import trtllm_rollout

        result["trtllm-device-uuid"] = _is_patched(trtllm_rollout.get_device_uuid)

    return result


def _failures(result: dict[str, Any]) -> list[str]:
    failures = []
    if result["numa-affinity"] is not True:
        failures.append("numa-affinity")
    if result["device-flops"] != {"C500": 240e12, "C550": 280e12}:
        failures.append("device-flops")
    if result["entropy-kernel-tile"] != [EXPECTED_TILE]:
        failures.append("entropy-kernel-tile")
    for identifier in ("megatron-asm-scopes-v1", "megatron-asm-scopes-v0"):
        if not all(result[identifier].values()):
            failures.append(identifier)
    for identifier in ("vllm-run-headless", "trtllm-device-uuid"):
        if result[identifier] not in (True, NOT_INSTALLED):
            failures.append(identifier)
    return failures


def _skips(result: dict[str, Any]) -> list[str]:
    return [
        identifier
        for identifier in ("vllm-run-headless", "trtllm-device-uuid")
        if result[identifier] == NOT_INSTALLED
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="auto", help="Ray address passed to ray.init")
    parser.add_argument("--num-gpus", type=float, default=1, help="GPU resources reserved by the probe worker")
    args = parser.parse_args()

    print("driver prepare:", prepare())
    ray.init(address=args.address)
    try:
        inspect_worker = ray.remote(num_gpus=args.num_gpus)(_inspect_worker)
        result = ray.get(inspect_worker.remote())
    finally:
        ray.shutdown()

    print("remote worker:")
    print(json.dumps(result, indent=2, sort_keys=True))
    failures = _failures(result)
    if failures:
        raise SystemExit(f"FAIL: workarounds missing in Ray worker: {', '.join(failures)}")
    skips = _skips(result)
    if skips:
        print(f"SKIP: optional dependencies not installed: {', '.join(skips)}")
    print("PASS: all workarounds with installed dependencies are active in the Ray worker")


if __name__ == "__main__":
    main()
