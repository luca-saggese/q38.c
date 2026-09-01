#!/usr/bin/env python3
"""Compare an independent semantic trace with the native q38 trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


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


def vector_check(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right):
        return {"status": "fail", "reason": "length", "left": len(left),
                "right": len(right)}
    errors = [abs(a - b) for a, b in zip(left, right)]
    return {
        "status": "pass" if all(close(a, b) for a, b in zip(left, right))
        else "fail",
        "max_abs": max(errors, default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads(args.native.read_text())
    reference = json.loads(args.reference.read_text())
    result = {
        "format": "q38-m6-semantic-comparison-v2",
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
    stage_keys = sorted(
        set(native_stages) | set(reference_stages),
        key=lambda key: (
            0 if key.startswith("layer:") else 1,
            int(key.split(":", 1)[1]),
            key.split(":", 1)[0],
        ),
    )
    for key in stage_keys:
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
        coordinate_mismatches = []
        for field in ("min", "max", "mean", "rms", "max_abs"):
            if not close(left_stats.get(field), right_stats.get(field)):
                mismatches.append(field)
        for field in ("min_index", "max_index", "max_abs_index"):
            if left_stats.get(field) != right_stats.get(field):
                coordinate_mismatches.append(field)
        for field in ("finite_count", "nan_count", "inf_count"):
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
        check["coordinate_mismatches"] = coordinate_mismatches
        check["checksum_equal"] = (
            left_stats.get("checksum") == right_stats.get("checksum")
        )
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
        if not right_weights:
            continue
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
    native_moe = native.get("layer2_moe_trace")
    ref_moe = reference.get("layer2_moe_trace")
    if native_moe is None or ref_moe is None:
        result["status"] = "fail"
        result.setdefault("first_divergence", "layer2_moe_trace")
        result["checks"].append(
            {"stage": "layer2_moe_trace", "status": "fail",
             "reason": "missing layer-2 boundary trace"}
        )
    else:
        moe_check = {"stage": "layer2_moe_trace", "status": "pass",
                     "vectors": {}}
        for field in (
            "router_input",
            "router_logits_pre_cast",
            "router_logits_effective",
            "selected_weights_pre_cast",
            "selected_weights_effective",
            "routed_output",
        ):
            check = vector_check(native_moe.get(field, []),
                                 ref_moe.get(field, []))
            moe_check["vectors"][field] = check
            if check["status"] != "pass":
                moe_check["status"] = "fail"
        native_rank = [item.get("expert") for item in native_moe.get(
            "top15_rank", [])]
        ref_rank = [item.get("expert") for item in ref_moe.get(
            "top15_rank", [])]
        if native_rank != ref_rank:
            moe_check["status"] = "fail"
            moe_check["top15_rank"] = {"status": "fail",
                                       "native": native_rank,
                                       "reference": ref_rank}
        if not close(native_moe.get("margin_rank10_rank11"),
                     ref_moe.get("margin_rank10_rank11")):
            moe_check["status"] = "fail"
            moe_check["margin_rank10_rank11"] = {
                "status": "fail",
                "native": native_moe.get("margin_rank10_rank11"),
                "reference": ref_moe.get("margin_rank10_rank11"),
            }
        result["checks"].append(moe_check)
        if moe_check["status"] != "pass":
            result["status"] = "fail"
            result.setdefault("first_divergence", "layer2_moe_trace")
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
