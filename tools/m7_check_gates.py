#!/usr/bin/env python3
"""Validate the currently committed M7 baseline artifacts."""

import json
from pathlib import Path


ROOT = Path("artifacts/m7")
REQUIRED = (
    "baseline_platform.json",
    "baseline_memory.json",
    "baseline_bench.json",
    "decode_profile.json",
)


def main() -> int:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("M7 gate failure: missing " + ", ".join(missing))

    bench = json.loads((ROOT / "baseline_bench.json").read_text())
    profile = json.loads((ROOT / "decode_profile.json").read_text())
    telemetry = bench["telemetry"]
    if bench["format"] != "q38-m7-bench-v1":
        raise SystemExit("M7 gate failure: invalid benchmark format")
    if bench["prompt_tokens"] != 1 or bench["decode_tokens"] != 1:
        raise SystemExit("M7 gate failure: baseline is not one-token")
    if bench["argmax"] != 11:
        raise SystemExit("M7 gate failure: greedy baseline changed")
    if bench["backend_rows"] <= 0 or bench["scalar_rows"] != 0:
        raise SystemExit("M7 gate failure: unexpected matvec backend mix")
    if bench["backend_declines"] != 0:
        raise SystemExit("M7 gate failure: backend declined rows")
    if profile["backend_rows"] != bench["backend_rows"]:
        raise SystemExit("M7 gate failure: profile/backend row mismatch")
    names = [record["name"] for record in telemetry["subsystems"]]
    if names != ["gdn", "qsa", "moe", "ple", "lm_head"]:
        raise SystemExit("M7 gate failure: subsystem schema mismatch")
    if telemetry["allocation_count"] < 0 or telemetry["cuda_synchronizations"] < 1:
        raise SystemExit("M7 gate failure: invalid allocator/synchronization telemetry")
    print("M7 baseline gates passed")
    return 0


if __name__ == "__main__":
    main()
