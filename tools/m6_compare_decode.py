#!/usr/bin/env python3
"""Fail-closed comparison of native and independent greedy decode evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads(args.native.read_text())
    oracle = json.loads(args.oracle.read_text())
    native_steps = native.get("steps", [])
    if oracle.get("format") == "q38-m6-stateful-gguf-oracle-v2":
        oracle_steps = (
            oracle.get("prompt_steps", []) + oracle.get("steps", [])
        )
    else:
        oracle_steps = oracle.get("steps", [])
    native_tokens = native.get("generated", [])
    oracle_tokens = oracle.get("generated", [])
    checks = []
    status = "pass"
    reason = None
    if oracle.get("status") != "pass":
        status = "blocked"
        reason = oracle.get("reason") or "independent oracle did not complete"
    elif not native_steps or not oracle_steps:
        status = "blocked"
        reason = "native and oracle step evidence is incomplete"
    else:
        count = min(len(native_steps), len(oracle_steps))
        for index in range(count):
            left, right = native_steps[index], oracle_steps[index]
            ok = (
                left.get("input_token") == right.get("input_token")
                and left.get("next_token") == right.get("next_token")
            )
            checks.append({
                "step": index,
                "native_input_token": left.get("input_token"),
                "oracle_input_token": right.get("input_token"),
                "native_next_token": left.get("next_token"),
                "oracle_next_token": right.get("next_token"),
                "status": "pass" if ok else "fail",
            })
            if not ok:
                status = "fail"
                if reason is None:
                    reason = (
                        f"step {index} token mismatch: native "
                        f"{left.get('next_token')} vs oracle "
                        f"{right.get('next_token')}"
                    )
        if len(native_steps) != len(oracle_steps):
            status = "fail"
            reason = "native/oracle step counts differ"
        if (
            native_steps[0].get("generated") is not False
            or native_steps[0].get("input_token") != oracle_steps[0].get("input_token")
            or native_steps[0].get("next_token") != oracle_steps[0].get("next_token")
        ):
            status = "fail"
            reason = reason or "fresh seed step is not equal"
    result = {
        "format": "q38-m6-decode-comparison-v2",
        "native": str(args.native),
        "oracle": str(args.oracle),
        "checks": checks,
        "fresh_seed": checks[0] if checks else None,
        "c12_fresh_comparison": (
            oracle.get("fresh_session", {}).get("comparison")
            if oracle.get("format") == "q38-m6-stateful-gguf-oracle-v2"
            else oracle.get("fresh_comparison")
        ),
        "native_generated": native_tokens,
        "oracle_generated": oracle_tokens,
        "status": status,
        "reason": reason,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
