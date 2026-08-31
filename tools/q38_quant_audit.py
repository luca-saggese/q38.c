#!/usr/bin/env python3
"""Audit runtime GGUF tensor types against the selective manifest."""

import argparse
import json
import re
import struct
from collections import defaultdict

TYPE_INFO = {
    8: (32, 34, "Q8_0"), 10: (256, 84, "Q2_K"), 12: (256, 144, "Q4_K"),
    16: (256, 66, "IQ2_XXS"), 27: (1, 8, "I64"), 30: (1, 2, "BF16"),
}


def string(stream):
    length = struct.unpack("<Q", stream.read(8))[0]
    value = stream.read(length)
    if len(value) != length:
        raise ValueError("truncated GGUF string")
    return value.decode()


def skip_value(stream, kind):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
             10: 8, 11: 8, 12: 8}
    if kind in sizes:
        stream.seek(sizes[kind], 1)
    elif kind == 8:
        string(stream)
    else:
        raise ValueError(f"unsupported metadata type {kind}")


def read(path):
    with open(path, "rb") as stream:
        magic, version, count, n_kv = struct.unpack("<IIQQ", stream.read(24))
        if magic != 0x46554747 or version != 3:
            raise ValueError("not GGUF v3")
        metadata = {}
        for _ in range(n_kv):
            key = string(stream)
            kind = struct.unpack("<I", stream.read(4))[0]
            if kind == 8:
                metadata[key] = string(stream)
            elif kind == 4:
                metadata[key] = struct.unpack("<I", stream.read(4))[0]
            elif kind == 7:
                metadata[key] = bool(struct.unpack("<B", stream.read(1))[0])
            else:
                skip_value(stream, kind)
        tensors = []
        for _ in range(count):
            name = string(stream)
            ndim = struct.unpack("<I", stream.read(4))[0]
            shape = list(struct.unpack("<" + "Q" * ndim, stream.read(8 * ndim)))
            kind, offset = struct.unpack("<IQ", stream.read(12))
            tensors.append((name, shape, kind, offset))
        data_start = (stream.tell() + 31) & ~31
        file_size = stream.seek(0, 2)
        result = []
        for name, shape, kind, offset in tensors:
            block, block_bytes, type_name = TYPE_INFO.get(kind, (0, 0, "unknown"))
            elements = 1
            for dim in shape:
                elements *= dim
            bytes_count = ((elements + block - 1) // block) * block_bytes if block else 0
            result.append({"name": name, "shape": shape, "type": type_name,
                           "elements": elements, "bytes": bytes_count,
                           "abs_offset": data_start + offset})
        return metadata, result, file_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--classes", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metadata, tensors, file_size = read(args.artifact)
    classes = {t["name"]: t for t in json.load(open(args.classes, encoding="utf-8"))["tensors"]}
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    rules = [(r, re.compile(r["pattern"])) for r in manifest["rules"]]
    histogram = defaultdict(lambda: defaultdict(lambda: {"tensors": 0, "bytes": 0}))
    violations = []
    for tensor in tensors:
        source = classes.get(tensor["name"])
        if not source:
            violations.append(f"{tensor['name']}: absent from source inventory")
            continue
        matches = [r for r, p in rules if r["class"] == source["class"] and p.search(tensor["name"])]
        if len(matches) != 1:
            violations.append(f"{tensor['name']}: manifest rule mismatch")
            continue
        rule = matches[0]
        expected = rule["quant_type"]
        if source["source_dtype"] != "BF16":
            expected = source["source_dtype"]
        elif expected in {"Q2_K", "IQ2_XXS", "Q4_K", "Q8_0"}:
            shape = list(source["shape"])
            if rule.get("layout_transform") == "transpose_last_two_axes":
                shape[-1], shape[-2] = shape[-2], shape[-1]
            if len(shape) < 2 or shape[-1] % TYPE_INFO[
                    {"Q2_K": 10, "IQ2_XXS": 16, "Q4_K": 12, "Q8_0": 8}[expected]][0]:
                expected = rule.get("fallback_quant_type", expected)
        if expected == "I64":
            expected = "I64"
        if expected != tensor["type"] and rule["include"]:
            violations.append(f"{tensor['name']}: expected {expected}, got {tensor['type']}")
        entry = histogram[source["class"]][tensor["type"]]
        entry["tensors"] += 1
        entry["bytes"] += tensor["bytes"]
    report = {"format": "q38_quant_audit_v1", "artifact": args.artifact,
              "metadata": metadata, "file_bytes": file_size,
              "classes": {k: dict(v) for k, v in sorted(histogram.items())},
              "violations": violations, "status": "pass" if not violations else "fail"}
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if violations:
        raise SystemExit("\n".join(violations))


if __name__ == "__main__":
    main()
