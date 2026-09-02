#!/usr/bin/env python3
"""Greedy autoregressive oracle over the quantized GGUF.

Each step recomputes the complete causal prefix with the independent GGUF
reader.  This is intentionally slower than a cache-aware implementation, but
it is stateful at the semantic level: the next token and state evidence are
derived from the exact prefix that precedes it, without importing q38 or
calling the native forward path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/home/lvx/q38model"))
    parser.add_argument("--prompt", type=str, default="9419")
    parser.add_argument("--generated-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.generated_count < 1 or args.generated_count > 128:
        raise SystemExit("generated-count must be in [1,128]")
    prompt = [int(value) for value in args.prompt.split(",") if value.strip()]
    if not prompt or any(value < 0 or value >= 248320 for value in prompt):
        raise SystemExit("prompt contains an invalid token ID")

    reference = Path(__file__).with_name("m6_quant_matched_reference.py")
    work = args.output.parent / ".m6_autoregressive_oracle"
    work.mkdir(parents=True, exist_ok=True)
    prefix = list(prompt)
    steps = []
    status = "pass"
    reason = None
    try:
        for step in range(args.generated_count):
            trace = work / "prefix_trace.json"
            report = work / "prefix_reference.json"
            trace.write_text(json.dumps({"tokens": prefix}))
            command = [
                sys.executable, str(reference), "--gguf", str(args.gguf),
                "--model-dir", str(args.model_dir), "--trace", str(trace),
                "--output", str(report), "--tokens",
                ",".join(str(value) for value in prefix),
                "--device", args.device,
            ]
            try:
                subprocess.run(command, check=True, timeout=args.timeout)
                data = json.loads(report.read_text())
                top = data["stages"][-1]["top"][0]
                token = int(top["id"])
                steps.append({
                    "step": step, "input_token": prefix[-1],
                    "next_token": token, "prefix_length": len(prefix),
                    "logits": data["stages"][-1]["stats"],
                })
                prefix.append(token)
            except (subprocess.SubprocessError, OSError, KeyError, IndexError,
                    ValueError) as exc:
                status = "blocked"
                reason = f"reference step {step} did not complete: {exc}"
                break
    finally:
        shutil.rmtree(work, ignore_errors=True)

    result = {
        "format": "q38-m6-autoregressive-oracle-v1",
        "reference": "independent GGUF reader/dequantizer + Qwen4Exp math",
        "gguf": str(args.gguf), "prompt": prompt,
        "requested_generated_count": args.generated_count,
        "generated": [entry["next_token"] for entry in steps],
        "steps": steps,
        "state_model": "exact causal-prefix recomputation; no q38 import",
        "status": status,
        "reason": reason,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
