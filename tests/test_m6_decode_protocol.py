#!/usr/bin/env python3
"""Regression checks for prompt prediction versus generated emission/consumption."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def check(path: Path, expected: list[int]) -> None:
    report = json.loads(path.read_text())
    if report.get("prompt") != [9419]:
        raise SystemExit(f"{path}: prompt is not [9419]")
    if report.get("generated") != expected:
        raise SystemExit(
            f"{path}: expected generated {expected}, got {report.get('generated')}"
        )
    steps = report.get("steps", [])
    if len(steps) != len(expected) + 1:
        raise SystemExit(f"{path}: trace length does not match protocol")
    prompt, emit = steps[:2]
    if (
        prompt.get("kind") != "prompt_prediction"
        or prompt.get("input_token") != 9419
        or prompt.get("next_token") != 11
        or prompt.get("state_committed") is not True
        or prompt.get("committed_tokens") != 1
    ):
        raise SystemExit(f"{path}: prompt prediction trace is incorrect")
    if (
        emit.get("kind") != "generated_emit"
        or emit.get("emitted_token") != 11
        or emit.get("consumed_token") != 4294967295
        or emit.get("state_committed") is not False
        or emit.get("committed_tokens") != 1
    ):
        raise SystemExit(f"{path}: first generated token was falsely committed")
    if len(expected) == 2:
        consume = steps[2]
        if (
            consume.get("kind") != "generated_consume"
            or consume.get("consumed_token") != 11
            or consume.get("emitted_token") != 353
            or consume.get("state_committed") is not True
            or consume.get("committed_tokens") != 2
        ):
            raise SystemExit(f"{path}: second generated token protocol is incorrect")


def main() -> None:
    paths = [Path(value) for value in sys.argv[1:]]
    if len(paths) != 2:
        raise SystemExit(
            "usage: test_m6_decode_protocol.py count1.json count2.json"
        )
    check(paths[0], [11])
    check(paths[1], [11, 353])
    print("test_m6_decode_protocol: prompt [9419] emission protocol passed")


if __name__ == "__main__":
    main()
