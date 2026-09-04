#!/usr/bin/env python3
"""Generate the deterministic token-sequence corpus used by M8 quality runs."""

import argparse
import hashlib
import json
from pathlib import Path


def sequences(count):
    state = 0x4D38514
    for index in range(count):
        length = 1 + index % 8
        values = []
        for _ in range(length):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            values.append(state % 248320)
        yield {
            "id": f"m8-seq-{index:04d}",
            "token_ids": values,
            "source": "deterministic LCG token probe; no generated output",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 32:
        raise SystemExit("M8 quality corpus requires at least 32 sequences")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for item in sequences(args.count):
            stream.write(json.dumps(item, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
