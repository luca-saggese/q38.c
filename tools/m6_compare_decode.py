#!/usr/bin/env python3
"""Fail-closed comparison of native and independent greedy decode evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def event_kind(step: dict) -> str:
    kind = step.get("kind")
    if kind:
        return kind
    return "generated_consume" if step.get("generated") else "prompt_prediction"


def event_tokens(step: dict, kind: str) -> tuple[object, object]:
    def optional(value: object) -> object:
        return None if value in (None, 4294967295) else value

    if kind == "generated_emit":
        return optional(step.get("consumed_token")), (
            step.get("emitted_token", step.get("next_token"))
        )
    if kind == "generated_consume":
        return (
            step.get("consumed_token", step.get("input_token")),
            step.get("emitted_token", step.get("next_token")),
        )
    return step.get("input_token"), step.get("next_token")


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
            left_kind, right_kind = event_kind(left), event_kind(right)
            left_input, left_output = event_tokens(left, left_kind)
            right_input, right_output = event_tokens(right, right_kind)
            ok = (
                left_kind == right_kind
                and left_input == right_input
                and left_output == right_output
            )
            checks.append({
                "step": index,
                "native_kind": left_kind,
                "oracle_kind": right_kind,
                "native_consumed_token": left_input,
                "oracle_consumed_token": right_input,
                "native_emitted_token": left_output,
                "oracle_emitted_token": right_output,
                "status": "pass" if ok else "fail",
            })
            if not ok:
                status = "fail"
                if reason is None:
                    reason = (
                        f"step {index} event mismatch: native "
                        f"{left_kind}/{left_input}/{left_output} vs oracle "
                        f"{right_kind}/{right_input}/{right_output}"
                    )
        if len(native_steps) != len(oracle_steps):
            status = "fail"
            reason = "native/oracle step counts differ"
        if native_tokens != oracle_tokens:
            status = "fail"
            reason = reason or "native/oracle generated token lists differ"
        if (
            event_kind(native_steps[0]) != "prompt_prediction"
            or event_kind(oracle_steps[0]) != "prompt_prediction"
            or event_tokens(native_steps[0], "prompt_prediction") !=
            event_tokens(oracle_steps[0], "prompt_prediction")
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
        "generated_equal": native_tokens == oracle_tokens,
        "status": status,
        "reason": reason,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
