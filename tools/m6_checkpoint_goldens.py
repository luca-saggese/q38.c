#!/usr/bin/env python3
"""Emit small, independent checkpoint-derived forward input goldens.

This deliberately records tensor metadata and the first BF16 values only.
It does not claim logits or a continuation, and it does not import q38.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def tensor_prefix(model_dir: Path, name: str, count: int = 16):
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard = model_dir / index["weight_map"][name]
    with shard.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        tensor = header[name]
        begin, end = tensor["data_offsets"]
        stream.seek(8 + header_size + begin)
        raw = stream.read(min(end - begin, count * 2))
    values = [struct.unpack("<H", raw[i : i + 2])[0] for i in range(0, len(raw), 2)]
    return {
        "name": name,
        "shard": shard.name,
        "shape": tensor["shape"],
        "dtype": tensor["dtype"],
        "first_bf16_bits": values,
        "prefix_sha256": hashlib.sha256(raw).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = ("model.language_model.embed_tokens.weight", "lm_head.weight")
    report = {
        "format": "q38_m6_checkpoint_golden_v1",
        "source": "local safetensors checkpoint",
        "claims": ["checkpoint-derived tensor prefixes", "no logits", "no continuation"],
        "tensors": [tensor_prefix(args.model_dir, name) for name in names],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
