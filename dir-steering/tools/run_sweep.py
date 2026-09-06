#!/usr/bin/env python3
"""Run a Q38 directional-steering scale sweep through the direct CLI."""

import argparse
import subprocess
from pathlib import Path


def prompts(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values:
        raise SystemExit(f"{path}: no prompts found")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q38", default="./q38")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--scales", default="-2,-1,-0.5,0,0.5,1,2")
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--attn-scale", type=float, default=0.0)
    args = parser.parse_args()

    for prompt in prompts(Path(args.prompts)):
        print("=" * 80)
        print(f"PROMPT: {prompt}")
        for scale in (float(value) for value in args.scales.split(",")):
            print("-" * 80)
            print(f"FFN scale: {scale:g}")
            subprocess.run(
                [
                    args.q38, "--generate", args.model, "--tokenizer",
                    args.tokenizer, "--ctx", str(args.ctx), "--max-tokens",
                    str(args.tokens), "--dir-steering-file", args.direction,
                    "--dir-steering-ffn", str(scale), "--dir-steering-attn",
                    str(args.attn_scale), "--prompt", prompt,
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
