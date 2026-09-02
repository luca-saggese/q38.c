#!/usr/bin/env python3
"""Compare CUDA diagnostic checkpoints against the scalar trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def stages(report: dict) -> dict[tuple[str, int], dict]:
    return {
        (entry.get("stage", ""), int(entry.get("layer", 0))): entry
        for entry in report.get("stages", [])
    }


def compare_stage(cpu: dict, gpu: dict) -> tuple[bool, float]:
    cstats, gstats = cpu.get("stats", {}), gpu.get("stats", {})
    maximum = 0.0
    for field in ("min", "max", "mean", "rms", "max_abs"):
        left, right = cstats.get(field), gstats.get(field)
        if left is None or right is None:
            if left != right:
                return False, math.inf
        else:
            maximum = max(maximum, abs(float(left) - float(right)))
            if not math.isclose(float(left), float(right), rel_tol=2e-3,
                                 abs_tol=2e-2):
                return False, maximum
    for field in ("finite_count", "nan_count", "inf_count"):
        if cstats.get(field) != gstats.get(field):
            return False, maximum
    return True, maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu = json.loads(args.cpu.read_text())
    gpu = json.loads(args.gpu.read_text())
    cpu_stages, gpu_stages = stages(cpu), stages(gpu)
    checkpoints = [("layer", 0), ("layer", 3), ("layer", 47),
                   ("final_norm", 0), ("logits", 0)]
    checks = []
    status = "pass"
    for key in checkpoints:
        left, right = cpu_stages.get(key), gpu_stages.get(key)
        if left is None or right is None:
            checks.append({"stage": f"{key[0]}:{key[1]}",
                           "status": "fail", "reason": "missing"})
            status = "fail"
            continue
        ok, maximum = compare_stage(left, right)
        checks.append({"stage": f"{key[0]}:{key[1]}",
                       "status": "pass" if ok else "fail",
                       "max_abs": maximum})
        if not ok:
            status = "fail"
    args.output.write_text(json.dumps({
        "format": "q38-m6-gpu-progressive-v1",
        "cpu": str(args.cpu),
        "gpu": str(args.gpu),
        "checkpoints": ["1-layer", "4-layer", "48-layer"],
        "checks": checks,
        "status": status,
    }, indent=2) + "\n")
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
