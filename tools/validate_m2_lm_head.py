#!/usr/bin/env python3
"""Validate the LM-head probe rows and ranking against frozen BF16 source."""

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
    name = "lm_head.weight"
    shard = root / index["weight_map"][name]
    with safe_open(str(shard), framework="pt") as source:
        rows = source.get_slice(name)
        hidden = torch.tensor(
            [(i % 31 - 15) * 0.0078125 + (i % 7) * 0.00003125 for i in range(2560)],
            dtype=torch.float32,
        )
        logits = []
        for entry in probe["rows"]:
            row = rows[entry["token_id"]]
            raw = row.contiguous().view(torch.uint16).cpu().numpy().tobytes()
            actual = fnv1a(raw)
            if actual != entry["fnv1a64"]:
                raise SystemExit(
                    f"LM-head token {entry['token_id']} mismatch: "
                    f"expected {actual}, got {entry['fnv1a64']}"
                )
            logits.append((row.float() * hidden).sum().item())
        token_ids = [entry["token_id"] for entry in probe["rows"]]
        expected_order = [
            token for _, token in sorted(zip(logits, token_ids), reverse=True)
        ]
        if expected_order != probe["top_order"]:
            raise SystemExit(
                f"LM-head ranking mismatch: expected {expected_order}, "
                f"got {probe['top_order']}"
            )
    print("validate_m2_lm_head: frozen BF16 rows and top-order match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
