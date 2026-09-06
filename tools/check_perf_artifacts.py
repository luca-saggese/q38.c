#!/usr/bin/env python3
"""Reject performance artifacts that cannot be compared under PERF_SCHEMA_V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCHEMA = "PERF_SCHEMA_V2"
REQUIRED = (
    "timer_schema_version",
    "execution_path",
    "model_fingerprint",
    "quant_recipe",
    "comparison_key",
)


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: cannot parse JSON: {exc}"]
    if not isinstance(document, dict):
        return [f"{path}: top-level value must be an object"]
    if document.get("timer_schema_version") != SCHEMA:
        errors.append(
            f"{path}: timer_schema_version must be {SCHEMA}"
        )
    for key in REQUIRED:
        if key not in document:
            errors.append(f"{path}: missing required field {key}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directories",
        nargs="*",
        type=Path,
        default=[Path("artifacts/perf/current"),
                 Path("artifacts/perf/reference_0")],
    )
    args = parser.parse_args()
    files = [
        path
        for directory in args.directories
        if directory.exists()
        for path in sorted(directory.glob("*.json"))
    ]
    errors = [error for path in files for error in check_file(path)]
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"PERF_SCHEMA_V2 artifact check passed ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
