#!/usr/bin/env python3
"""Check prefix token/state equality across CUDA decode ladder artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [(path, json.loads(path.read_text())) for path in args.reports]
    checks = []
    status = "pass"
    reason = None
    if not reports:
        status, reason = "blocked", "no ladder reports supplied"
    else:
        baseline = reports[0][1].get("steps", [])
        for path, report in reports[1:]:
            steps = report.get("steps", [])
            count = min(len(baseline), len(steps))
            for index in range(count):
                left, right = baseline[index], steps[index]
                ok = (
                    left.get("next_token") == right.get("next_token")
                    and left.get("logits_hash") == right.get("logits_hash")
                    and left.get("gdn_state_hash") == right.get("gdn_state_hash")
                    and left.get("conv_history_hash") ==
                    right.get("conv_history_hash")
                    and left.get("ple_history_hash") ==
                    right.get("ple_history_hash")
                )
                checks.append({
                    "report": str(path), "step": index,
                    "status": "pass" if ok else "fail",
                })
                if not ok:
                    status = "fail"
    result = {
        "format": "q38-m6-decode-ladder-check-v1",
        "reports": [str(path) for path, _ in reports],
        "checks": checks, "status": status, "reason": reason,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
