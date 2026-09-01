#!/usr/bin/env python3
"""Inspect whether the repository can truthfully run all 48 text layers."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config = json.loads((args.model_dir / "config.json").read_text())
    text = config["text_config"]
    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    layers = []
    for layer in range(text["num_hidden_layers"]):
        prefix = f"model.language_model.layers.{layer}."
        names = [n for n in index if n.startswith(prefix)]
        has_gdn = any(".linear_attn." in n for n in names)
        has_qsa = any(".self_attn." in n for n in names)
        has_moe = any(".mlp.experts." in n for n in names)
        layers.append({"layer": layer, "gdn": has_gdn, "qsa": has_qsa, "moe": has_moe})
    source = Path("q38_forward.c").read_text()
    has_full = "q38_forward_full" in source
    has_decode = Path("q38_decode.c").exists()
    report = {
        "status": "pass" if has_full and has_decode else "fail",
        "model_type": config["architectures"][0],
        "layers": layers,
        "checkpoint_bytes": sum(
            (args.model_dir / name).stat().st_size
            for name in set(index.values())
        ),
        "full_forward_api": has_full,
        "decode_api": has_decode,
        "reason": None if has_full and has_decode else
            "q38 has layer-level QSA and MoE primitives but no complete "
            "48-layer forward/decode implementation; no logits are claimed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
