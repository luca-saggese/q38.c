#!/usr/bin/env python3
"""Fail-closed validation of native/official/C12 seed alignment."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_m6_decode_seed_state.py artifact.json")
    report = json.loads(Path(sys.argv[1]).read_text())
    if report.get("format") != "q38-m6-decode-seed-state-v1":
        raise SystemExit("seed-state format mismatch")
    if report.get("status") != "pass":
        raise SystemExit("seed-state evidence is not passing")
    fresh = report.get("fresh_seed", {})
    native = fresh.get("native", {})
    oracle = fresh.get("official_oracle", {})
    c12 = fresh.get("c12", {})
    if (
        native.get("input_token") != 9419
        or oracle.get("input_token") != 9419
        or native.get("next_token") != 11
        or oracle.get("next_token") != 11
        or c12.get("status") != "pass"
    ):
        raise SystemExit("fresh seed is not equal across native, oracle, and C12")
    continuation = report.get("first_generated", {})
    if (
        continuation.get("native", {}).get("next_token") != 353
        or continuation.get("official_oracle", {}).get("next_token") != 353
    ):
        raise SystemExit("first generated continuation is not equal")
    print("test_m6_decode_seed_state: native, official oracle, and C12 agree")


if __name__ == "__main__":
    main()
