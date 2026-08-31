#!/usr/bin/env python3
"""Assemble deterministic M1 acceptance artifacts without hiding blockers."""

import argparse
import hashlib
import json
import os


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)

    source = json.load(open(args.inventory, encoding="utf-8"))
    classes = json.load(open(args.classes, encoding="utf-8"))
    excluded = [t for t in classes["tensors"] if not t["included_runtime"]]
    runtime = [t for t in classes["tensors"] if t["included_runtime"]]
    with open(os.path.join(args.artifact_dir, "runtime_inventory.json"), "w",
              encoding="utf-8") as stream:
        json.dump({"format": "q38_runtime_inventory_v1",
                   "tensor_count": len(runtime), "tensors": runtime},
                  stream, indent=2, sort_keys=True)
    with open(os.path.join(args.artifact_dir, "excluded_tensors.json"), "w",
              encoding="utf-8") as stream:
        json.dump({"tensors": excluded,
                   "summary": classes["excluded_summary"]}, stream,
                  indent=2, sort_keys=True)
        stream.write("\n")

    files = [args.inventory, args.classes, args.manifest]
    artifact = os.path.join(
        args.artifact_dir,
        "qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf")
    subset = os.path.join(
        args.artifact_dir,
        "qwen38-runtime-only-layers0-3-Q2Experts-BF16Core-BF16PLE.gguf")
    if os.path.exists(artifact):
        files.append(artifact)
    if os.path.exists(subset):
        files.append(subset)
    with open(os.path.join(args.artifact_dir, "checksums.txt"), "w",
              encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{digest(path)}  {path}\n")

    memory_files = [os.path.join(args.artifact_dir, "memory_cold.json"),
                    os.path.join(args.artifact_dir, "memory_warm.json")]
    blockers = []
    if not os.path.exists(artifact):
        blockers.append("full 48-layer runtime artifact is not present")
    for path in memory_files:
        if not os.path.exists(path):
            blockers.append(f"missing memory artifact: {path}")
    report = {
        "format": "q38_m1_acceptance_v1",
        "source_tensor_count": source["tensor_count"],
        "runtime_tensor_count": classes["tensor_count"] -
        classes["excluded_summary"]["tensors"],
        "excluded_summary": classes["excluded_summary"],
        "status": "blocked" if blockers else "pass",
        "blockers": blockers,
        "whole_file_cuda_host_register": False,
        "persistent_dequant_mirror": False,
    }
    with open(os.path.join(args.artifact_dir, "acceptance.txt"), "w",
              encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if blockers:
        print("M1 acceptance gated: " + "; ".join(blockers))
    else:
        print("M1 acceptance passed")


if __name__ == "__main__":
    main()
