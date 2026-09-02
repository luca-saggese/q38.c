#!/usr/bin/env python3
"""Fail closed if the classified M6 boundary divergence returns."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(
        "artifacts/m6/quant_matched_comparison.json"
    )
    report = json.loads(path.read_text())
    progressive = next(
        check for check in report.get("checks", [])
        if check.get("stage") == "progressive_boundaries"
    )
    assert progressive.get("status") == "pass", (
        "progressive boundary comparison diverged: "
        + repr(progressive.get("first_divergence"))
    )
    assert progressive.get("first_divergence") is None
    for layer in progressive.get("checks", []):
        assert layer.get("status") == "pass", layer
        assert layer.get("first_divergence") is None, layer
        if layer.get("layer") == 1:
            ple_norm = next(
                check for check in layer.get("checks", [])
                if check.get("name") == "ple_key_normed"
            )
            assert ple_norm.get("status") == "pass", ple_norm
    print("test_m6_progressive_boundaries: classified boundary gates pass")


if __name__ == "__main__":
    main()
