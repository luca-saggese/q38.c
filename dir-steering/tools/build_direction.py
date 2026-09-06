#!/usr/bin/env python3
"""Build a Q38 directional-steering vector from paired prompt sets."""

import argparse
import array
import json
import math
import subprocess
import tempfile
from pathlib import Path

N_LAYER = 48
N_EMBD = 2560


def read_prompts(path: Path) -> list[str]:
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        raise SystemExit(f"{path}: no prompts found")
    return prompts


def normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if not norm or not math.isfinite(norm):
        raise RuntimeError("cannot normalize a zero/non-finite direction")
    return [value / norm for value in values]


def capture(q38: Path, model: Path, tokenizer: Path, prompt: str,
            component: str, ctx: int, work: Path) -> list[list[float]]:
    dump = work / "dump"
    dump.mkdir()
    subprocess.run(
        [
            str(q38), "--generate", str(model), "--tokenizer", str(tokenizer),
            "--prompt", prompt, "--ctx", str(ctx), "--max-tokens", "1",
            "--dump-steering-dir", str(dump),
            "--dump-steering-component", component,
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    rows = []
    for layer in range(N_LAYER):
        path = dump / f"{component}-{layer}-pos0.bin"
        data = array.array("f")
        with path.open("rb") as stream:
            data.fromfile(stream, path.stat().st_size // 4)
        if len(data) != N_EMBD:
            raise RuntimeError(f"{path}: expected {N_EMBD} floats, got {len(data)}")
        rows.append(list(data))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q38", default="./q38")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--good-file", required=True)
    parser.add_argument("--bad-file", required=True)
    parser.add_argument("--out", default="dir-steering/out/direction.json")
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--component", choices=("ffn_out", "attn_out"),
                        default="ffn_out")
    args = parser.parse_args()

    q38 = Path(args.q38).resolve()
    model = Path(args.model).resolve()
    tokenizer = Path(args.tokenizer).resolve()
    good = read_prompts(Path(args.good_file))
    bad = read_prompts(Path(args.bad_file))
    count = min(len(good), len(bad))
    good, bad = good[:count], bad[:count]
    sums = [[0.0] * N_EMBD for _ in range(N_LAYER)]

    with tempfile.TemporaryDirectory(prefix="q38-dir-steer-") as temp:
        root = Path(temp)
        for index, (good_prompt, bad_prompt) in enumerate(zip(good, bad), 1):
            good_rows = capture(q38, model, tokenizer, good_prompt,
                                args.component, args.ctx, root / f"good-{index}")
            bad_rows = capture(q38, model, tokenizer, bad_prompt,
                               args.component, args.ctx, root / f"bad-{index}")
            for layer in range(N_LAYER):
                for column in range(N_EMBD):
                    sums[layer][column] += (
                        good_rows[layer][column] - bad_rows[layer][column]
                    )

    directions = [normalize(row) for row in sums]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "q38-directional-steering-v1",
        "shape": [N_LAYER, N_EMBD],
        "component": args.component,
        "pairs": count,
        "model": str(model),
        "note": "positive scale suppresses this direction; negative amplifies it",
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    flat = array.array("f", (value for row in directions for value in row))
    binary = output.with_suffix(".f32")
    with binary.open("wb") as stream:
        flat.tofile(stream)
    print(f"wrote {output}")
    print(f"wrote {binary}")


if __name__ == "__main__":
    main()
