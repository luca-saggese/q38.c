#!/usr/bin/env python3
"""Gate for a separate quant-matched reference execution.

This command deliberately does not invoke q38.  A llama.cpp executable may be
provided as a separate reference artifact; otherwise the gate remains blocked
instead of treating native q38 output as an independent reference.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama", type=Path)
    args = parser.parse_args()
    report = {
        "format": "q38-m6-quant-matched-reference-v1",
        "reference": "separate llama.cpp GGUF reader/dequantizer",
        "gguf": str(args.gguf),
        "input_trace": str(args.trace),
        "q38_forward_called": False,
        "status": "blocked",
    }
    executable = args.llama
    if executable is None:
        found = shutil.which("llama-cli")
        executable = Path(found) if found else None
    if executable is None or not executable.exists():
        report["reason"] = (
            "blocked: no separate llama.cpp executable is available, and an "
            "independent GGUF reader/dequantizer alone cannot produce a "
            "quant-matched 48-layer Qwen4Exp trace without reimplementing the "
            "full forward graph; native q38 output cannot serve as reference"
        )
    else:
        report["reason"] = (
            "a separate llama.cpp runner was found but this gate requires its "
            "explicit model-specific trace adapter before claiming statistics"
        )
        report["llama"] = str(executable)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
