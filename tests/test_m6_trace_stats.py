#!/usr/bin/env python3
"""Regression checks for finite-only trace statistics."""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.m6_transformers_reference import stats


def main() -> None:
    result = stats(torch.tensor([1.0, float("nan"), -2.0, float("inf")]))
    assert result["finite_count"] == 2
    assert result["nan_count"] == 1
    assert result["inf_count"] == 1
    assert result["min_index"] == 2
    assert result["max_index"] == 0
    assert result["max_abs_index"] == 2
    assert math.isclose(result["mean"], -0.5)
    assert math.isclose(result["rms"], math.sqrt(2.5), rel_tol=1e-6)

    empty = stats(torch.tensor([float("nan"), float("-inf")]))
    assert empty["finite_count"] == 0
    assert empty["min"] is None
    assert empty["mean"] is None
    assert empty["min_index"] is None
    print("test_m6_trace_stats: finite-only statistics passed")


if __name__ == "__main__":
    main()
