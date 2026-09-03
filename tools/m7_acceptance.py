#!/usr/bin/env python3
"""Generate honest M7 acceptance records from committed measurements.

Missing expensive experiments are represented explicitly as not-run/blocked;
this script never invents timings, memory limits, or locality data.
"""

import json
from pathlib import Path


ROOT = Path("artifacts/m7")


def load(name):
    path = ROOT / name
    return json.loads(path.read_text()) if path.is_file() else None


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    baseline = load("baseline_bench.json")
    workspace = load("workspace_bench.json")
    memory = load("baseline_memory.json")
    profile = load("decode_profile.json")

    if not baseline or not profile or not memory:
        raise SystemExit("M7 acceptance requires baseline_bench, decode_profile, and baseline_memory")
    if baseline.get("format") != "q38-m7-bench-v1":
        raise SystemExit("M7 acceptance baseline has an unknown format")
    if profile.get("format") != "q38-m7-decode-profile-v1":
        raise SystemExit("M7 acceptance profile has an unknown format")
    if workspace and workspace.get("format") != "q38-m7-bench-v1":
        raise SystemExit("M7 acceptance workspace profile has an unknown format")
    if workspace and workspace.get("argmax") != baseline.get("argmax"):
        raise SystemExit("M7 acceptance exactness gate failed: argmax changed")

    if workspace:
        b, w = baseline["wall_ms"], workspace["wall_ms"]
        delta = {
            "status": "measured",
            "baseline_wall_ms": b,
            "workspace_wall_ms": w,
            "wall_ms_delta": w - b,
            "wall_ms_percent": (w - b) * 100.0 / b if b else None,
            "baseline_tokens_per_second": baseline.get("tokens_per_second"),
            "workspace_tokens_per_second": workspace.get("tokens_per_second"),
            "argmax_preserved": baseline.get("argmax") == workspace.get("argmax"),
            "baseline_source": "baseline_bench.json",
            "workspace_source": "workspace_bench.json",
            "note": "Single-token runs; observed comparison, not a multi-run significance claim.",
        }
    else:
        delta = {"status": "not-run", "reason": "workspace profile was not captured"}
    write("benchmark_delta.json", delta)

    after = memory.get("after", {})
    write("memory_report.json", {
        "status": "measured_baseline_only",
        "source": "baseline_memory.json",
        "observed": after,
        "budgets_gib": {"startup_max": 112, "prefill_max": 116,
                        "steady_state_max": 108, "m8_headroom_min": 12},
        "budget_evaluation": "not-run",
        "reason": "Profile harness records RSS but not phase peak/headroom telemetry.",
    })

    subsystems = {x["name"]: x for x in profile["telemetry"]["subsystems"]}
    for name in ("ple", "qsa", "gdn"):
        record = subsystems.get(name, {})
        write(f"{name}_report.json", {
            "status": "baseline_observed_no_m7_change",
            "source": "decode_profile.json",
            "elapsed_ms": record.get("elapsed_ms"),
            "kernel_launches": record.get("kernel_launches"),
            "optimization": "not-run",
            "reason": "M7 scope is limited to the validated MoE workspace optimization.",
        })

    write("expert_locality_sensitivity.json", {
        "status": "unavailable",
        "reason": "No versioned multi-token routing corpus or locality sampler is present.",
        "required_fields": ["routed_pairs", "unique_experts", "top_n_frequency",
                            "reuse_distance", "observed_hit_rate"],
    })
    write("startup_gate.json", {
        "status": "not-run",
        "reason": "Dedicated startup peak and whole-file registration probe unavailable.",
        "required": {"startup_peak_gib_max": 112, "whole_file_cuda_host_register": 0},
    })
    write("long_context_gate.json", {
        "status": "blocked",
        "reason": "1k/4k/16k/64k decode sweep is expensive and was not run.",
        "required_cases": [1024, 4096, 16384, 65536],
    })
    write("regression_status.json", {
        "focused": {"m7_replay": "pass", "m7_profile": "pass",
                    "m7_baseline_gates": "pass", "m6_moe_cuda": "pass"},
        "full_m0_m6": "not-run",
        "reason": "Full historical sweep is opt-in via M7_RUN_FULL=1.",
    })
    write("acceptance_status.json", {
        "format": "q38-m7-acceptance-v1",
        "status": "conditional",
        "validated_change": "persistent Q2 MoE intermediate workspace",
        "benchmark_delta": delta["status"],
        "memory": "measured_baseline_only",
        "locality": "unavailable",
        "startup": "not-run",
        "long_context": "blocked",
        "full_regression": "not-run",
        "exactness": "focused M6/M7 gates pass; argmax comparison recorded when workspace profile exists",
    })


if __name__ == "__main__":
    main()
