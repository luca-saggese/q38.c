#!/usr/bin/env python3
"""Independent M7 gate for decode and backend telemetry schemas."""

import json
import sys
from pathlib import Path


SUBSYSTEMS = ("gdn", "qsa", "moe", "ple", "lm_head")
SUBSYSTEM_RECORD_FIELDS = (
    "name",
    "callbacks",
    "kernel_launches",
    "synchronizations",
    "allocation_count",
    "allocation_bytes",
    "elapsed_ms",
    "weight_bytes",
    "activation_read_bytes",
    "activation_write_bytes",
    "d2h_bytes",
    "host_syncs",
    "effective_weight_gbps",
)
BENCH_FIELDS = (
    "format",
    "case",
    "prompt_tokens",
    "decode_tokens",
    "runs",
    "wall_ms",
    "tokens_per_second",
    "argmax",
    "memory",
    "telemetry",
    "matrix_calls",
    "backend_rows",
    "scalar_rows",
    "backend_declines",
    "row_matvec_dequant_ms",
    "launch_overhead_proxy_ms",
    "unattributed_gpu_ms",
)
PROFILE_FIELDS = (
    "format",
    "classification",
    "matrix_calls",
    "backend_rows",
    "scalar_rows",
    "backend_declines",
    "row_matvec_dequant_ms",
    "launch_overhead_proxy_ms",
    "unattributed_gpu_ms",
    "telemetry",
)
BENCH_NUMERIC_FIELDS = (
    "wall_ms",
    "tokens_per_second",
    "matrix_calls",
    "backend_rows",
    "scalar_rows",
    "backend_declines",
    "row_matvec_dequant_ms",
    "launch_overhead_proxy_ms",
    "unattributed_gpu_ms",
)
PROFILE_NUMERIC_FIELDS = (
    "matrix_calls",
    "backend_rows",
    "scalar_rows",
    "backend_declines",
    "row_matvec_dequant_ms",
    "launch_overhead_proxy_ms",
    "unattributed_gpu_ms",
)
TELEMETRY_FIELDS = (
    "version",
    "token_count",
    "weight_bytes",
    "activation_read_bytes",
    "activation_write_bytes",
    "kernel_ms",
    "effective_weight_gbps",
    "host_syncs",
    "d2h_bytes",
    "weight_bytes_per_token",
    "kernel_ms_per_token",
    "host_syncs_per_token",
    "d2h_bytes_per_token",
    "cuda_elapsed_ms",
    "cuda_synchronizations",
    "allocation_count",
    "allocation_bytes",
    "subsystems",
)
TELEMETRY_RECORD_FIELDS = (
    "subsystem", "layer", "logical_stage", "tensor_name", "qtype",
    "rows", "cols", "bytes", "resident_hit", "resident_miss",
    "upload_bytes", "weight_bytes", "activation_read_bytes",
    "activation_write_bytes", "d2h_bytes", "upload_ms", "kernel_ms",
    "backend_overhead_ms", "allocation_count", "sync_count", "host_syncs",
)


def require_fields(value, fields, label):
    missing = [field for field in fields if field not in value]
    if missing:
        raise AssertionError(f"{label} missing fields: {', '.join(missing)}")


def nonnegative(value, label):
    if not isinstance(value, (int, float)) or value < 0:
        raise AssertionError(f"{label} must be a non-negative number")


def check_telemetry(telemetry, label):
    require_fields(telemetry, ("version", "cuda_elapsed_ms",
                               "cuda_synchronizations", "allocation_count",
                               "allocation_bytes", "subsystems"), label)
    if telemetry["version"] not in (1, 2):
        raise AssertionError(f"{label}.version must be 1 or 2")
    numeric = ["cuda_elapsed_ms", "cuda_synchronizations",
               "allocation_count", "allocation_bytes"]
    if telemetry["version"] == 2:
        require_fields(telemetry, TELEMETRY_FIELDS[1:-1], label)
        numeric += list(TELEMETRY_FIELDS[1:-1])
    for field in numeric:
        nonnegative(telemetry[field], f"{label}.{field}")
    records = telemetry["subsystems"]
    if not isinstance(records, list):
        raise AssertionError(f"{label}.subsystems must be an array")
    if [record.get("name") for record in records] != list(SUBSYSTEMS):
        raise AssertionError(f"{label}.subsystems names do not match M7 schema")
    for index, record in enumerate(records):
        base_fields = ("name", "callbacks", "kernel_launches",
                       "synchronizations", "allocation_count",
                       "allocation_bytes", "elapsed_ms")
        require_fields(record, base_fields, f"{label}.subsystems[{index}]")
        for field in base_fields[1:]:
            nonnegative(record[field], f"{label}.subsystems[{index}].{field}")
        if telemetry["version"] == 2:
            for field in ("weight_bytes", "activation_read_bytes",
                          "activation_write_bytes", "d2h_bytes", "host_syncs",
                          "effective_weight_gbps"):
                nonnegative(record[field], f"{label}.subsystems[{index}].{field}")
    for index, record in enumerate(telemetry.get("records", [])):
        fields = TELEMETRY_RECORD_FIELDS if telemetry["version"] == 2 else (
            "subsystem", "layer", "logical_stage", "tensor_name", "qtype",
            "rows", "cols", "bytes", "resident_hit", "resident_miss",
            "upload_bytes", "upload_ms", "kernel_ms",
            "backend_overhead_ms", "allocation_count", "sync_count")
        require_fields(record, fields, f"{label}.records[{index}]")
        for field in ("layer", "qtype", "rows", "cols", "bytes",
                      "upload_bytes", "upload_ms", "kernel_ms",
                      "backend_overhead_ms", "allocation_count", "sync_count"):
            nonnegative(record[field], f"{label}.records[{index}].{field}")
        if not isinstance(record["resident_hit"], bool) or not isinstance(
                record["resident_miss"], bool):
            raise AssertionError(f"{label}.records[{index}] residency flags must be bool")


def load(root, name):
    path = root / name
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AssertionError(f"missing {path}") from exc
    except json.JSONDecodeError as exc:
        raise AssertionError(f"invalid JSON in {path}: {exc}") from exc


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/m7")
    bench = load(root, "baseline_bench.json")
    profile = load(root, "decode_profile.json")
    require_fields(bench, BENCH_FIELDS, "baseline_bench")
    require_fields(profile, PROFILE_FIELDS, "decode_profile")
    if bench["format"] != "q38-m7-bench-v1":
        raise AssertionError("baseline_bench.format is not q38-m7-bench-v1")
    if profile["format"] != "q38-m7-decode-profile-v1":
        raise AssertionError("decode_profile.format is not q38-m7-decode-profile-v1")
    check_telemetry(bench["telemetry"], "baseline_bench.telemetry")
    check_telemetry(profile["telemetry"], "decode_profile.telemetry")

    for field in BENCH_NUMERIC_FIELDS:
        nonnegative(bench[field], f"baseline_bench.{field}")
    for field in PROFILE_NUMERIC_FIELDS:
        nonnegative(profile[field], f"decode_profile.{field}")
    for field in ("matrix_calls", "backend_rows", "scalar_rows", "backend_declines"):
        if bench[field] != profile[field]:
            raise AssertionError(f"{field} differs between benchmark and profile")
    print("M7 independent profile schema gates passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        raise SystemExit(f"M7 profile schema gate failure: {error}")
