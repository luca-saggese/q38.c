#!/usr/bin/env python3
"""Compare an independent semantic trace with the native q38 trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


STAGES = ("layer:0", "layer:3", "layer:7", "layer:15", "layer:31",
          "layer:47", "final_norm:0", "logits:0")
ATOL = 0.02
RTOL = 0.002


def stage_map(report: dict) -> dict[str, dict]:
    return {
        f"{stage['stage']}:{stage.get('layer', 0)}": stage
        for stage in report.get("stages", [])
    }


def close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a == b
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads(args.native.read_text())
    reference = json.loads(args.reference.read_text())
    result = {
        "format": "q38-m6-semantic-comparison-v1",
        "native": str(args.native),
        "reference": str(args.reference),
        "tokens": native.get("tokens"),
        "tolerance": {"absolute": ATOL, "relative": RTOL},
        "checks": [],
        "status": "pass",
    }
    if native.get("tokens") != reference.get("tokens"):
        result["status"] = "fail"
        result["first_divergence"] = "input_tokens"
    native_stages = stage_map(native)
    reference_stages = stage_map(reference)
    for key in STAGES:
        left, right = native_stages.get(key), reference_stages.get(key)
        check = {"stage": key, "status": "pass"}
        if left is None or right is None:
            check.update(status="fail", reason="missing full-vector stage")
            result["status"] = "fail"
            result.setdefault("first_divergence", key)
            result["checks"].append(check)
            continue
        left_stats, right_stats = left.get("stats"), right.get("stats")
        if not left_stats or not right_stats:
            check.update(status="fail", reason="full-vector statistics missing")
            result["status"] = "fail"
            result.setdefault("first_divergence", key)
            result["checks"].append(check)
            continue
        mismatches = []
        for field in ("min", "max", "mean", "rms", "max_abs"):
            if not close(left_stats.get(field), right_stats.get(field)):
                mismatches.append(field)
        for field in ("nan_count", "inf_count"):
            if left_stats.get(field) != right_stats.get(field):
                mismatches.append(field)
        left_fixed = left_stats.get("fixed", left.get("fixed", []))
        right_fixed = right_stats.get("fixed", right.get("fixed", []))
        if len(left_fixed) != len(right_fixed):
            mismatches.append("fixed_coordinates")
        else:
            for lcoord, rcoord in zip(left_fixed, right_fixed):
                if lcoord.get("index") != rcoord.get("index") or not close(
                    lcoord.get("value"), rcoord.get("value")
                ):
                    mismatches.append(f"fixed[{lcoord.get('index')}]")
        check["mismatches"] = mismatches
        if mismatches:
            check["status"] = "fail"
            result["status"] = "fail"
            result.setdefault("first_divergence", key)
        result["checks"].append(check)

    native_top = native_stages.get("logits", {}).get("top")
    ref_top = reference_stages.get("logits", {}).get("top")
    if native_top is None or ref_top is None or [
        item.get("id") for item in native_top
    ] != [item.get("id") for item in ref_top]:
        result["status"] = "fail"
        result.setdefault("first_divergence", "logits:top_k")
        result["checks"].append(
            {"stage": "logits:top_k", "status": "fail", "reason": "argmax/top-k"}
        )
    native_routes = native.get("routing", [])
    ref_routes = reference.get("routing", [])
    route_mismatch = []
    if len(native_routes) != len(ref_routes):
        route_mismatch.append("route_count")
    for left, right in zip(native_routes, ref_routes):
        left_experts = left.get("experts", [])
        right_experts = right.get("experts", [])
        if left_experts and isinstance(left_experts[0], list):
            left_experts = left_experts[0]
        if right_experts and isinstance(right_experts[0], list):
            right_experts = right_experts[0]
        if left_experts != right_experts:
            route_mismatch.append(f"layer:{left.get('layer')}:experts")
        left_weights = left.get("weights", [])
        right_weights = right.get("weights", [])
        if left_weights and isinstance(left_weights[0], list):
            left_weights = left_weights[0]
        if right_weights and isinstance(right_weights[0], list):
            right_weights = right_weights[0]
        if len(left_weights) != len(right_weights):
            route_mismatch.append(f"layer:{left.get('layer')}:weights")
        elif any(
            not close(lvalue, rvalue)
            for lvalue, rvalue in zip(left_weights, right_weights)
        ):
            route_mismatch.append(f"layer:{left.get('layer')}:weights")
    if route_mismatch:
        result["status"] = "fail"
        result.setdefault("first_divergence", "routing")
        result["checks"].append(
            {"stage": "routing", "status": "fail", "reason": "expert decisions",
             "mismatches": route_mismatch}
        )
    if native.get("qsa_selection") != reference.get("qsa_selection"):
        result["status"] = "fail"
        result.setdefault("first_divergence", "qsa_selection")
        result["checks"].append(
            {"stage": "qsa_selection", "status": "fail",
             "reason": "QSA decisions"}
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
