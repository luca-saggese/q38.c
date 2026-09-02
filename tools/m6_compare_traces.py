#!/usr/bin/env python3
"""Compare an independent semantic trace with the native q38 trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ATOL = 0.02
RTOL = 0.002
LAYER7_MAX_ABS = 0.10
LAYER7_RMS = 0.01
LAYER7_RELATIVE_RMS = 0.15
LAYER7_COSINE = 0.99


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


def vector_metrics(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right):
        return {"status": "fail", "reason": "length"}
    errors = [a - b for a, b in zip(left, right)]
    left_rms = math.sqrt(sum(a * a for a in left) / len(left))
    right_rms = math.sqrt(sum(b * b for b in right) / len(right))
    error_rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    cosine = (
        sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
        if left_norm and right_norm else None
    )
    extreme = sorted(
        range(len(errors)), key=lambda i: abs(errors[i]), reverse=True
    )[:4]
    return {
        "status": "pass",
        "max_abs": max((abs(e) for e in errors), default=0.0),
        "rms": error_rms,
        "relative_rms": error_rms / right_rms if right_rms else None,
        "cosine_similarity": cosine,
        "extreme_coordinates": [
            {"index": i, "native": left[i], "reference": right[i],
             "delta": errors[i]}
            for i in extreme
        ],
    }


def route_values(route: dict) -> tuple[list[int], list[float]]:
    experts = route.get("experts", [])
    weights = route.get("weights", [])
    if experts and isinstance(experts[0], list):
        experts = experts[0]
    if weights and isinstance(weights[0], list):
        weights = weights[0]
    return experts, weights


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
        "hard_gates": {},
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
        if key == "layer:7":
            left_values, right_values = left.get("values"), right.get("values")
            if left_values is not None and right_values is not None:
                check["vector_metrics"] = vector_metrics(
                    left_values, right_values
                )
                metrics = check["vector_metrics"]
                layer7_pass = (
                    metrics["max_abs"] <= LAYER7_MAX_ABS
                    and metrics["rms"] <= LAYER7_RMS
                    and metrics["relative_rms"] <= LAYER7_RELATIVE_RMS
                    and metrics["cosine_similarity"] >= LAYER7_COSINE
                )
                check["status"] = "pass" if layer7_pass else "fail"
                result["hard_gates"]["layer7_hidden"] = {
                    "status": "pass" if layer7_pass else "fail",
                    "tolerance": {
                        "max_abs": LAYER7_MAX_ABS,
                        "rms": LAYER7_RMS,
                        "relative_rms": LAYER7_RELATIVE_RMS,
                        "cosine_similarity": LAYER7_COSINE,
                    },
                }
                if not layer7_pass:
                    result["status"] = "fail"
                    result.setdefault("first_divergence", key)
            elif mismatches:
                check["status"] = "fail"
                result["hard_gates"]["layer7_hidden"] = {
                    "status": "fail", "reason": "missing full layer-7 vectors"
                }
                result["status"] = "fail"
                result.setdefault("first_divergence", key)
        elif mismatches:
            check["status"] = "diagnostic"
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
    route_diagnostics = []
    if len(native_routes) != len(ref_routes):
        route_mismatch.append("route_count")
    for left, right in zip(native_routes, ref_routes):
        left_experts, left_weights = route_values(left)
        right_experts, right_weights = route_values(right)
        layer = left.get("layer")
        if set(left_experts) != set(right_experts):
            route_mismatch.append(f"layer:{layer}:selected_set")
        elif left_experts != right_experts:
            route_diagnostics.append(f"layer:{layer}:selected_order")
        if not right_weights:
            continue
        if len(left_weights) != len(right_weights):
            route_mismatch.append(f"layer:{layer}:weights")
        else:
            left_by_expert = dict(zip(left_experts, left_weights))
            right_by_expert = dict(zip(right_experts, right_weights))
            if set(left_by_expert) != set(right_by_expert) or any(
                not close(left_by_expert[expert], right_by_expert[expert])
                for expert in right_by_expert
            ):
                route_mismatch.append(f"layer:{layer}:weights_by_expert")
    if route_mismatch or route_diagnostics:
        if route_mismatch:
            result["status"] = "fail"
        if route_mismatch:
            result.setdefault("first_divergence", "routing")
        result["checks"].append(
            {"stage": "routing",
             "status": "fail" if route_mismatch else "pass",
             "selected_set_exact": not any(
                 item.endswith(":selected_set") for item in route_mismatch
             ),
             "selected_order_exact": not route_diagnostics,
             "mismatches": route_mismatch,
             "diagnostics": route_diagnostics}
        )
        result["hard_gates"]["routing_selected_set"] = {
            "status": "fail" if route_mismatch else "pass",
            "selected_set_exact": not any(
                item.endswith(":selected_set") for item in route_mismatch
            ),
            "weights_by_expert_checked": True,
        }
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
        moe_check = {"stage": "layer2_moe_trace", "status": "diagnostic",
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
            moe_check["top15_rank"] = {"status": "fail",
                                       "native": native_rank,
                                       "reference": ref_rank}
        if not close(native_moe.get("margin_rank10_rank11"),
                     ref_moe.get("margin_rank10_rank11")):
            moe_check["margin_rank10_rank11"] = {
                "status": "diagnostic",
                "native": native_moe.get("margin_rank10_rank11"),
                "reference": ref_moe.get("margin_rank10_rank11"),
            }
        result["checks"].append(moe_check)
        native_selected = native_moe.get("selected_experts", [])
        ref_selected = ref_moe.get("selected_experts", [])
        if set(native_selected) != set(ref_selected):
            result["status"] = "fail"
            result.setdefault("first_divergence", "layer2_moe_trace")
        else:
            moe_check["selected_set_exact"] = True
            moe_check["selected_order_exact"] = native_selected == ref_selected
            native_weights = dict(zip(
                native_selected, native_moe.get("selected_weights_effective", [])
            ))
            ref_weights = dict(zip(
                ref_selected, ref_moe.get("selected_weights_effective", [])
            ))
            moe_check["weights_by_expert_exact"] = (
                set(native_weights) == set(ref_weights) and all(
                    close(native_weights[e], ref_weights[e]) for e in ref_weights
                )
            )
            routed = vector_check(
                native_moe.get("routed_output", []),
                ref_moe.get("routed_output", []),
            )
            moe_check["routed_weighted_sum"] = routed
            if not moe_check["weights_by_expert_exact"] or routed["status"] != "pass":
                result["status"] = "fail"
                result.setdefault("first_divergence", "layer2_moe_trace")
            elif moe_check["weights_by_expert_exact"] and routed["status"] == "pass":
                moe_check["status"] = "pass"
        result["hard_gates"]["layer2_selected_set"] = {
            "status": "pass" if moe_check.get("selected_set_exact") else "fail",
            "selected_order_exact": moe_check.get("selected_order_exact", False),
            "weights_by_expert_exact": moe_check.get(
                "weights_by_expert_exact", False
            ),
            "routed_weighted_sum": moe_check.get("routed_weighted_sum"),
        }
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
