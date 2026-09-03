#!/usr/bin/env python3
"""Project M8 recipe memory from real tensor inventory and quant manifest."""

import argparse
import hashlib
import json
import re
from pathlib import Path


BLOCK_INFO = {
    "Q2_K": (256, 84),
    "IQ2_XXS": (256, 66),
    "Q4_K": (256, 144),
    "Q8_0": (32, 34),
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantized_bytes(tensor, target, fallback, layout_transform=None):
    shape = list(tensor["shape"])
    if layout_transform == "transpose_last_two_axes":
        shape[-1], shape[-2] = shape[-2], shape[-1]
    effective = target
    if target in BLOCK_INFO and (len(shape) < 2 or
                                 shape[-1] % BLOCK_INFO[target][0]):
        effective = fallback or tensor["source_dtype"]
    elements = 1
    for dimension in shape:
        elements *= dimension
    if effective == "BF16":
        return elements * 2, effective
    if effective == "I64":
        return elements * 8, effective
    if effective not in BLOCK_INFO:
        raise ValueError(f"{tensor['name']}: unsupported quant type {effective}")
    block, block_bytes = BLOCK_INFO[effective]
    return (elements // shape[-1]) * (shape[-1] // block) * block_bytes, effective


def solve(classes, manifest, alignment):
    rules = [(rule, re.compile(rule["pattern"])) for rule in manifest["rules"]]
    rows = []
    for tensor in classes["tensors"]:
        if not tensor.get("included_runtime", True):
            continue
        matches = [(rule, pattern) for rule, pattern in rules
                   if rule["class"] == tensor["class"]
                   and pattern.search(tensor["name"])]
        if len(matches) != 1:
            raise ValueError(f"{tensor['name']}: expected one manifest rule")
        rule = matches[0][0]
        target = rule["quant_type"] if tensor["source_dtype"] == "BF16" else tensor["source_dtype"]
        payload, effective = quantized_bytes(
            tensor, target, rule.get("fallback_quant_type"),
            rule.get("layout_transform"))
        rows.append({
            "name": tensor["name"],
            "class": tensor["class"],
            "source_bytes": tensor["bytes_source"],
            "quantized_bytes": payload,
            "alignment_overhead": (-payload) % alignment,
            "effective_type": effective,
        })
    by_class = {}
    for row in rows:
        item = by_class.setdefault(row["class"], {
            "tensors": 0, "source_bytes": 0, "quantized_bytes": 0,
            "alignment_overhead": 0, "effective_types": {},
        })
        item["tensors"] += 1
        item["source_bytes"] += row["source_bytes"]
        item["quantized_bytes"] += row["quantized_bytes"]
        item["alignment_overhead"] += row["alignment_overhead"]
        item["effective_types"][row["effective_type"]] = (
            item["effective_types"].get(row["effective_type"], 0) + 1
        )
    payload = sum(item["quantized_bytes"] for item in by_class.values())
    overhead = sum(item["alignment_overhead"] for item in by_class.values())
    return rows, by_class, payload, overhead


def measured_values(path):
    if not path:
        return {}
    data = load(path)
    after = data.get("after", data)
    return {
        "measured_peak_bytes": after.get("rss_bytes"),
        "measured_mmap_bytes": after.get("model_mapped_bytes"),
        "measured_model_file_bytes": after.get("model_file_bytes"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibrate-artifact")
    parser.add_argument("--measured-memory")
    parser.add_argument("--alignment", type=int, default=32)
    parser.add_argument("--residency-policy", default="all_non_ple",
                        choices=("full_file", "all_non_ple", "none"))
    parser.add_argument("--persistent-state-bytes", type=int, default=0)
    parser.add_argument("--workspace-bytes", type=int, default=0)
    parser.add_argument("--cache-budget-bytes", type=int, default=0)
    args = parser.parse_args()
    if args.alignment <= 0 or args.alignment & (args.alignment - 1):
        raise SystemExit("--alignment must be a positive power of two")

    classes = load(args.classes)
    manifest = load(args.manifest)
    rows, by_class, payload, data_overhead = solve(
        classes, manifest, args.alignment)
    calibration = {}
    if args.calibrate_artifact:
        measured_file = Path(args.calibrate_artifact).stat().st_size
        calibration = {
            "artifact": args.calibrate_artifact,
            "artifact_sha256": sha256(args.calibrate_artifact),
            "measured_file_bytes": measured_file,
            "estimated_payload_plus_alignment_bytes": payload + data_overhead,
            "fixed_header_bytes": measured_file - payload - data_overhead,
        }
        if calibration["fixed_header_bytes"] < 0:
            raise SystemExit("calibration artifact is smaller than estimated payload")
    fixed_header = calibration.get("fixed_header_bytes", 0)
    model_file = payload + data_overhead + fixed_header
    ple_bytes = by_class.get("ple", {}).get("quantized_bytes", 0)
    if args.residency_policy == "full_file":
        resident = model_file
    elif args.residency_policy == "all_non_ple":
        resident = model_file - ple_bytes
    else:
        resident = 0
    projected_peak = (resident + args.persistent_state_bytes +
                      args.workspace_bytes + args.cache_budget_bytes)
    measured = measured_values(args.measured_memory)
    peak_error = None
    if measured.get("measured_peak_bytes"):
        peak_error = ((projected_peak - measured["measured_peak_bytes"]) * 100.0 /
                      measured["measured_peak_bytes"])
    result = {
        "format": "q38-m8-memory-solver-v1",
        "classes": str(Path(args.classes)),
        "manifest": str(Path(args.manifest)),
        "manifest_sha256": sha256(args.manifest),
        "alignment": args.alignment,
        "components": by_class,
        "model_file_bytes": model_file,
        "mmap_bytes": model_file,
        "predicted_resident_bytes": resident,
        "persistent_state_bytes": args.persistent_state_bytes,
        "max_workspace_bytes": args.workspace_bytes,
        "cache_budget_bytes": args.cache_budget_bytes,
        "projected_peak_bytes": projected_peak,
        "residency_policy": args.residency_policy,
        "calibration": calibration or {"status": "not-calibrated"},
        "measured": measured,
        "peak_error_percent": peak_error,
        "gates": {
            "startup_max_bytes": 112 * 1024**3,
            "prefill_max_bytes": 116 * 1024**3,
            "steady_state_preferred_bytes": 108 * 1024**3,
            "startup": "pass" if projected_peak <= 112 * 1024**3 else "fail",
            "prefill": "pass" if projected_peak <= 116 * 1024**3 else "fail",
            "steady_state": "pass" if projected_peak <= 108 * 1024**3 else "fail",
        },
        "status": "calibrated" if calibration else "projected",
        "tensor_rows": rows,
    }
    if peak_error is not None:
        result["calibration"]["peak_comparison"] = {
            "within_five_percent": abs(peak_error) <= 5.0,
            "error_percent": peak_error,
        }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
