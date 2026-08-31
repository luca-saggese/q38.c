#!/usr/bin/env python3
"""Extract a deterministic tensor inventory from a Safetensors checkpoint."""

import argparse
import json
import os
import struct
from collections import defaultdict


def read_header(path):
    with open(path, "rb") as stream:
        raw_len = stream.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: missing Safetensors header length")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len > 128 * 1024 * 1024:
            raise ValueError(f"{path}: unreasonable Safetensors header")
        raw_header = stream.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError(f"{path}: truncated Safetensors header")
    return json.loads(raw_header)


def inventory(model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as stream:
        index = json.load(stream)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")

    headers = {}
    for shard in sorted(set(weight_map.values())):
        headers[shard] = read_header(os.path.join(model_dir, shard))

    tensors = []
    seen = set()
    for name in sorted(weight_map):
        shard = weight_map[name]
        entry = headers[shard].get(name)
        if not isinstance(entry, dict) or "data_offsets" not in entry:
            raise ValueError(f"{name}: missing tensor entry in {shard}")
        if name in seen:
            raise ValueError(f"duplicate tensor: {name}")
        seen.add(name)
        offsets = entry["data_offsets"]
        if len(offsets) != 2 or offsets[1] < offsets[0]:
            raise ValueError(f"{name}: invalid data_offsets")
        shape = entry.get("shape")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or dim < 0 for dim in shape
        ):
            raise ValueError(f"{name}: invalid shape")
        elements = 1
        for dim in shape:
            elements *= dim
        tensors.append(
            {
                "name": name,
                "shape": shape,
                "gguf_type": None,
                "source_dtype": entry.get("dtype"),
                "elements": elements,
                "bytes_source": offsets[1] - offsets[0],
                "source_shard": shard,
            }
        )

    by_shard = defaultdict(lambda: {"tensors": 0, "bytes_source": 0})
    for tensor in tensors:
        summary = by_shard[tensor["source_shard"]]
        summary["tensors"] += 1
        summary["bytes_source"] += tensor["bytes_source"]
    return {
        "format": "q38_source_inventory_v1",
        "model_dir": os.path.abspath(model_dir),
        "tensor_count": len(tensors),
        "tensors": tensors,
        "shards": dict(sorted(by_shard.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/home/lvx/q38model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = inventory(args.model_dir)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
