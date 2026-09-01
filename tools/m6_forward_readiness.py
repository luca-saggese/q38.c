#!/usr/bin/env python3
"""Inspect whether the repository can truthfully run all 48 text layers."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight", type=Path,
                   default=Path("artifacts/m6/preflight.json"))
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
    preflight = json.loads(args.preflight.read_text())
    preflight_pass = preflight.get("status") == "pass"
    report = {
        "status": "pass" if has_full and has_decode and preflight_pass else "fail",
        "model_type": config["architectures"][0],
        "layers": layers,
        "checkpoint_bytes": sum(
            (args.model_dir / name).stat().st_size
            for name in set(index.values())
        ),
        "full_forward_api": has_full,
        "decode_api": has_decode,
        "preflight": str(args.preflight),
        "preflight_pass": preflight_pass,
        "first_token_logits_verified": False,
        "decode_verified": False,
        "reason": None if has_full and has_decode and preflight_pass else
            "complete 48-layer forward/decode APIs and a passing tensor "
            "preflight are required; no logits are claimed",
    }
    if report["status"] == "pass":
        report["reason"] = (
            "structural preflight passed; first-token hidden/logit goldens "
            "still require a live full-model execution")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
