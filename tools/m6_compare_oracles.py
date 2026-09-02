#!/usr/bin/env python3
"""Fail-closed CPU/GPU equality check for the official-cache oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_stats(left: dict, right: dict, tolerance: float) -> dict:
    fields = ("min", "max", "mean", "rms", "max_abs")
    deltas = {}
    for field in fields:
        a, b = left.get(field), right.get(field)
        deltas[field] = None if a is None or b is None else abs(float(a) - float(b))
    failed = [
        field for field, delta in deltas.items()
        if delta is not None and delta > tolerance
    ]
    return {
        "status": "pass" if not failed else "fail",
        "tolerance": tolerance,
        "deltas": deltas,
        "failed_fields": failed,
    }


def cache_shape(value: dict) -> dict:
    """Remove device and checksum fields while retaining official state shape."""
    if not value:
        return {}
    result = {
        "position_ids": {
            key: value.get("position_ids", {}).get(key)
            for key in ("present", "shape", "dtype")
        },
        "layers": [],
    }
    for layer in value.get("layers", []):
        entry = {
            "layer": layer.get("layer"),
            "type": layer.get("type"),
            "is_initialized": layer.get("is_initialized"),
            "has_previous_state": layer.get("has_previous_state", {}),
            "is_indexer_initialized": layer.get("is_indexer_initialized"),
        }
        for group in ("conv_states", "recurrent_states"):
            entry[group] = {
                key: {
                    subkey: item.get(subkey)
                    for subkey in ("present", "shape", "dtype")
                }
                for key, item in layer.get(group, {}).items()
            }
        for group in ("keys", "values", "indexer_keys"):
            item = layer.get(group, {})
            entry[group] = {
                key: item.get(key) for key in ("present", "shape", "dtype")
            }
        result["layers"].append(entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="CPU/GPU reduction tolerance; token/cache equality remains exact",
    )
    args = parser.parse_args()
    cpu = json.loads(args.cpu.read_text())
    gpu = json.loads(args.gpu.read_text())
    checks = []
    status = "pass"
    reason = None
    if cpu.get("status") != "pass" or gpu.get("status") != "pass":
        status = "blocked"
        reason = "CPU and GPU oracle reports must both be pass"
    elif cpu.get("tokens") != gpu.get("tokens"):
        status = "fail"
        reason = "CPU and GPU token sequences differ"
    else:
        cpu_steps, gpu_steps = cpu.get("steps", []), gpu.get("steps", [])
        if len(cpu_steps) != len(gpu_steps):
            status = "fail"
            reason = "CPU and GPU step counts differ"
        for index, (left, right) in enumerate(zip(cpu_steps, gpu_steps)):
            final_norm = compare_stats(
                left.get("final_norm", {}),
                right.get("final_norm", {}),
                args.tolerance,
            )
            logits = compare_stats(
                left.get("logits", {}),
                right.get("logits", {}),
                args.tolerance,
            )
            ok = (
                left.get("input_token") == right.get("input_token")
                and left.get("next_token") == right.get("next_token")
                and left.get("cache_seq_length") == right.get("cache_seq_length")
                and final_norm["status"] == "pass"
                and logits["status"] == "pass"
            )
            checks.append({
                "step": index,
                "input_token": left.get("input_token"),
                "next_token": left.get("next_token"),
                "final_norm": final_norm,
                "logits": logits,
                "status": "pass" if ok else "fail",
            })
            if not ok:
                status = "fail"
                reason = reason or f"CPU/GPU mismatch at step {index}"
    cache_equal = cache_shape(cpu.get("final_cache", {})) == cache_shape(
        gpu.get("final_cache", {})
    )
    if not cache_equal:
        status = "fail"
        reason = reason or "CPU and GPU official cache shapes/states differ"
    result = {
        "format": "q38-m6-oracle-cpu-gpu-comparison-v1",
        "cpu": str(args.cpu),
        "gpu": str(args.gpu),
        "tolerance": args.tolerance,
        "checks": checks,
        "cache_shape_equal": cache_equal,
        "cpu_device_check": cpu.get("device_check"),
        "gpu_device_check": gpu.get("device_check"),
        "status": status,
        "reason": reason,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
