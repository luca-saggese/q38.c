#!/usr/bin/env python3
"""Aggregate real CLI runs while preserving unavailable telemetry as null."""

import argparse
import json
import statistics
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = max(0, min(len(values) - 1,
                       int((len(values) - 1) * fraction + 0.999999)))
    return values[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m7-baseline", type=Path)
    args = parser.parse_args()
    paths = sorted(args.runs.glob("run_*.json"))
    if len(paths) != 6:
        raise SystemExit("expected exactly one cold and five warm run files")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    timings = [float(report["timing_ms"]["first_token"]) for report in reports]
    memories = [report.get("memory", {}) for report in reports]
    available = {
        "first_token_ms": timings,
        "peak_cuda_allocated_bytes": [
            memory.get("peak_cuda_allocated_bytes") for memory in memories
        ],
        "peak_rss_bytes": [memory.get("peak_rss_bytes") for memory in memories],
    }
    result = {
        "format": "q38-m8-r1-cold-warm-v2",
        "artifact": str(args.artifact),
        "ple_policy": {
            "storage": "file-backed SSD/mmap",
            "cache": "bounded",
            "staging": "bounded",
            "resident_bytes": "excluded from residency fit",
            "full_mirror": False,
        },
        "residency_fit": {
            "criterion": "non-PLE resident bytes only",
            "total_gguf_bytes_is_fit_criterion": False,
        },
        "runs": 6,
        "cold_runs": 1,
        "warm_runs": 5,
        "method": {
            "runtime_api": "q38 --generate",
            "prompt": reports[0].get("prompt"),
            "max_tokens": len(reports[0].get("generated_ids", [])),
            "cold_definition": "first process invocation",
            "warm_definition": "subsequent process invocations",
            "m7_same_process": False,
            "note": (
                "The repository is in a pre-existing CUDA merge-conflict "
                "state, so the same-process M7 harness could not be rebuilt. "
                "These are real R1 runs, not a same-process M7 claim."
            ),
        },
        "cold": {
            "wall_ms": timings[0],
            "peak_cuda_allocated_bytes": available[
                "peak_cuda_allocated_bytes"][0],
            "peak_rss_bytes": available["peak_rss_bytes"][0],
        },
        "warm": {
            "wall_ms": timings[1:],
            "median_ms": statistics.median(timings[1:]),
            "p95_ms": percentile(timings[1:], 0.95),
        },
        "observed": {
            "peak_cuda_allocated_bytes": max(
                available["peak_cuda_allocated_bytes"]
            ),
            "peak_rss_bytes": max(available["peak_rss_bytes"]),
            "cuda_total_bytes": memories[0].get("cuda_total_bytes"),
        },
        "required_m7_telemetry": {
            "resident_non_ple_bytes": None,
            "warm_median_ms": statistics.median(timings[1:]),
            "warm_p95_ms": percentile(timings[1:], 0.95),
            "cold_ms": timings[0],
            "cuda_unified_peak_bytes": None,
            "rss_bytes": max(available["peak_rss_bytes"]),
            "mem_available_bytes": None,
            "upload_bytes_per_token": None,
            "resident_misses": None,
            "warm_upload_bytes_per_token": None,
            "warm_resident_misses": None,
            "status": "not-computed",
            "reason": (
                "The existing --generate JSON API exposes workspace "
                "allocation and RSS only; it does not expose persistent "
                "non-PLE residency, MemAvailable, upload counters, or misses."
            ),
        },
        "status": "conditional",
    }
    if args.m7_baseline:
        baseline = json.loads(args.m7_baseline.read_text(encoding="utf-8"))
        baseline_memory = baseline.get("memory", {})
        result["direct_q2_m7_vs_q4_r1"] = {
            "q2_m7_artifact": (
                "artifacts/m1/"
                "qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf"
            ),
            "q2_m7_evidence": str(args.m7_baseline),
            "q2_m7_first_token_ms": baseline.get("timing_ms", {}).get(
                "first_token"
            ),
            "q2_m7_median_decode_ms": statistics.median(
                baseline.get("timing_ms", {}).get("per_token", [None])[1:]
            ) if len(baseline.get("timing_ms", {}).get("per_token", [])) > 1
            else None,
            "q2_m7_peak_rss_bytes": baseline_memory.get("peak_rss_bytes"),
            "q4_r1_cold_ms": timings[0],
            "q4_r1_warm_median_ms": statistics.median(timings[1:]),
            "q4_r1_warm_p95_ms": percentile(timings[1:], 0.95),
            "quality_drift_vs_bf16_reference": None,
            "quality_drift_status": "not-computed",
            "reason": (
                "The paired BF16/reference quality record set is not "
                "available; no drift is inferred from generated text."
            ),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
