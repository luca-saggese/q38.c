#!/usr/bin/env python3
"""Fail closed when the native M6 trace lacks routing diagnostics."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(
        "artifacts/m6/real_forward_trace.json"
    )
    report = json.loads(path.read_text())
    routing = {int(item["layer"]): item for item in report.get("routing", [])}
    router = {
        int(item["layer"]): item for item in report.get("router_logits", [])
    }
    assert set(routing) == set(range(48)), "routing diagnostics are incomplete"
    assert set(router) == set(range(48)), "router diagnostics are incomplete"
    for layer in range(48):
        route = routing[layer]
        assert len(route.get("experts", [])) == 10
        assert len(route.get("weights", [])) == 10
        assert len(route.get("hidden_input", [])) == 2560
        assert len(route.get("routed_output", [])) == 2560
        detail = router[layer]
        assert len(detail.get("logits", [])) == 512
        assert len(detail.get("top", [])) >= 11
        assert detail.get("rank10", {}).get("expert") is not None
        assert detail.get("rank11", {}).get("expert") is not None
        assert math.isfinite(float(detail["margin_rank10_rank11"]))
    print("test_m6_trace_schema: complete per-layer routing diagnostics passed")


if __name__ == "__main__":
    main()
