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
        "format": "q38-m8-r1-cold-warm-v1",
        "artifact": str(args.artifact),
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
            "status": "not-computed",
            "reason": (
                "The existing --generate JSON API exposes workspace "
                "allocation and RSS only; it does not expose persistent "
                "non-PLE residency, MemAvailable, upload counters, or misses."
            ),
        },
        "status": "conditional",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
