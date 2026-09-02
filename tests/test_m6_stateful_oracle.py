#!/usr/bin/env python3
"""Fail-closed checks for the official-cache stateful oracle artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_m6_stateful_oracle.py artifact.json")
    report = json.loads(Path(sys.argv[1]).read_text())
    if report.get("format") != "q38-m6-stateful-gguf-oracle-v2":
        raise SystemExit("stateful oracle format mismatch")
    if report.get("status") != "pass":
        raise SystemExit(f"stateful oracle is not passing: {report.get('reason')}")
    fresh = report.get("fresh_session", {})
    comparison = fresh.get("comparison", {})
    if fresh.get("status") != "pass" or comparison.get("status") != "pass":
        raise SystemExit("fresh-session comparison did not pass")
    if comparison.get("actual_token") != comparison.get("expected_token"):
        raise SystemExit("fresh-session token mismatch")
    checkpoints = comparison.get("checkpoints", {})
    required = {"0", "3", "7", "15", "31", "47"}
    if set(checkpoints) != required or any(
        item.get("status") != "pass" for item in checkpoints.values()
    ):
        raise SystemExit("checkpoint comparison is incomplete or failed")

    initial = fresh.get("initial_cache", {})
    layers = initial.get("layers", [])
    if len(layers) != 48:
        raise SystemExit("initial cache does not describe all decoder layers")
    if any(layer.get("is_initialized") for layer in layers):
        raise SystemExit("initial cache contains initialized decoder state")
    if layers[0].get("type") != "LinearAttentionLayer":
        raise SystemExit("layer 0 is not an official linear-attention cache")
    if layers[3].get("type") != "DynamicIndexedLayer":
        raise SystemExit("layer 3 is not an official indexed-attention cache")
    if any(
        evidence.get("present")
        for layer in layers
        for group in ("conv_states", "recurrent_states")
        for evidence in layer.get(group, {}).values()
    ):
        raise SystemExit("initial GDN/conv/PLE cache state is not empty")
    if not report.get("prompt_steps") or not report.get("checkpoints"):
        raise SystemExit("stateful oracle evidence is incomplete")
    if report.get("steps"):
        first = report["steps"][0]
        if first.get("input_token") != comparison.get("actual_token"):
            raise SystemExit("continuation does not consume the fresh argmax")
        if report.get("generated", [None])[0] != first.get("next_token"):
            raise SystemExit("generated token list disagrees with continuation trace")
    print("test_m6_stateful_oracle: fresh token, checkpoints, and cache state passed")


if __name__ == "__main__":
    main()
