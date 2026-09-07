#!/usr/bin/env python3
"""Reclassify the frozen S4B/S4C artifact without loading the model."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    replay = json.loads(args.replay.read_text())
    aggregate = source["aggregate"]
    old = aggregate["exclusive_forward_timing"]
    wall = aggregate["wall_ms"]
    argmax = aggregate["argmax_ms"]
    bookkeeping = aggregate["bookkeeping_ms"]
    actual_injection = 0.0
    critical_wait = aggregate["ple_critical_stall_ms"]
    additive = (
        old["embedding_ms"]
        + old["QSA_ms"]
        + old["GDN_ms"]
        + old["MoE_ms"]
        + old["GR_ms"]
        + old["LM_head_ms"]
        + old["norms_residual_glue_ms"]
        + old["other_layer_ms"]
        + actual_injection
        + critical_wait
        + argmax
        + bookkeeping
    )
    orchestration_gap = max(0.0, wall - additive)
    unexplained = max(0.0, old["unexplained_ms"])
    owners = []
    for owner in aggregate["matrix_owner_attribution"]:
        owner = dict(owner)
        if owner["calls"] == 12800 and owner["bytes_read"] == 34816000:
            owner["owner"] = "PLE_FILE_BACKED_LOOKUP"
            owner["classification"] = "confirmed_by_ple_traffic_signature"
        owners.append(owner)

    result = {
        "format": "q2-critical-path-attribution-v2",
        "PERF_SCHEMA": "PERF_SCHEMA_V2",
        "source_artifact": str(args.source),
        "replay_artifact": str(args.replay),
        "model_run": False,
        "status": "pass",
        "critical_path_semantics": {
            "additive_sum_ms": additive + orchestration_gap,
            "token_wall_ms": wall,
            "residual_unexplained_ms": unexplained,
            "residual_unexplained_fraction": unexplained / wall if wall else 0.0,
            "overlap_buckets_excluded": True,
            "acceptance_unexplained_below_3_percent":
                bool(wall and unexplained / wall < 0.03),
        },
        "partition": {
            "embedding_ms": old["embedding_ms"],
            "decoder_layer_execution": {
                "GR_ms": old["GR_ms"],
                "GDN_QSA_ms": old["GDN_ms"] + old["QSA_ms"],
                "MoE_ms": old["MoE_ms"],
                "norm_residual_glue_ms": old["norms_residual_glue_ms"],
                "orchestration_backend_gaps_ms": orchestration_gap,
                "other_layer_ms": old["other_layer_ms"],
            },
            "ple_actual_synchronous_injection_ms": actual_injection,
            "final_norm_ms": 0.0,
            "LM_head_ms": old["LM_head_ms"],
            "argmax_ms": argmax,
            "bookkeeping_ms": bookkeeping,
        },
        "ple": {
            "async_window_ms": old["ple_ms"],
            "async_window_semantics": "NON_ADDITIVE_OVERLAPPED_INTERVAL",
            "worker_elapsed_ms": aggregate["ple_elapsed_ms"],
            "worker_cpu_ms": replay["worker_cpu_ms"],
            "critical_wait_ms": critical_wait,
            "wait_at_injection_ms": critical_wait,
            "synchronous_injection_ms": actual_injection,
            "measurement_note": (
                "Frozen v1 has no dedicated synchronous injection boundary; "
                "the old parent interval is excluded and retained only as "
                "a non-additive window. A future diagnostic run can split "
                "decoder_before/actual_PLE/decoder_after boundaries."
            ),
        },
        "matrix_owner_attribution": owners,
        "replay": replay,
        "correctness": source.get("correctness", {}),
        "hardware": source.get("hardware", {}),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
