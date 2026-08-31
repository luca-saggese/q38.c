#!/usr/bin/env python3
"""Compare selected GGUF embedding rows with the frozen BF16 safetensors."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def fnv1a(data: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in data:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--probe", required=True)
    args = parser.parse_args()
    root = Path(args.model_dir)
    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    index = json.loads((root / "model.safetensors.index.json").read_text())
    name = "model.language_model.embed_tokens.weight"
    shard = root / index["weight_map"][name]
    with safe_open(str(shard), framework="pt") as source:
        rows = source.get_slice(name)
        for entry in probe["rows"]:
            row = rows[entry["token_id"]]
            raw = row.contiguous().view(torch.uint16).cpu().numpy().tobytes()
            actual = fnv1a(raw)
            if actual != entry["fnv1a64"]:
                raise SystemExit(
                    f"embedding token {entry['token_id']} mismatch: "
                    f"expected {actual}, got {entry['fnv1a64']}"
                )
    print("validate_m2_embedding: frozen BF16 rows match GGUF probe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
