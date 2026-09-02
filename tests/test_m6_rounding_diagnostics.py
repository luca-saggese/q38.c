#!/usr/bin/env python3
"""Ensure layer-9 rounding evidence is present without changing tolerances."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "artifacts/m6/quant_matched_comparison.json"
    )
    report = json.loads(path.read_text())
    routing = next(
        check for check in report.get("checks", [])
        if check.get("stage") == "routing"
    )
    layer9 = next(
        item for item in routing.get("per_layer", [])
        if item.get("layer") == 9
    )
    rounding = layer9["rounding_diagnostics"]
    for field in (
        "native_pre_cast_vs_reference",
        "native_effective",
        "native_vs_bf16_matmul",
        "native_vs_fp32_cast",
    ):
        assert rounding[field].get("status") in ("pass", "fail", "diagnostic")
    assert isinstance(rounding["native_effective_bits_exact"], bool)
    assert layer9["router_chain"]["native_present"]
    assert layer9["router_weight_rows"]["native_present"]
    print("test_m6_rounding_diagnostics: layer-9 rounding evidence present")


if __name__ == "__main__":
    main()
