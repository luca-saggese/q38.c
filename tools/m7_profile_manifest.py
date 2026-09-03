#!/usr/bin/env python3
"""Create immutable, content-addressed M7 profile records."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: m7_profile_manifest.py PROFILE_DIR OUTPUT_DIR")
    profile_dir, output_dir = map(Path, sys.argv[1:])
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    bench = json.loads((profile_dir / "baseline_bench.json").read_text())
    profile = json.loads((profile_dir / "decode_profile.json").read_text())
    record = {
        "format": "q38-m7-profile-manifest-v1",
        "commit_sha": commit,
        "binary_sha256": sha256(Path("q38")),
        "profile_sha256": sha256(profile_dir / "decode_profile.json"),
        "benchmark_sha256": sha256(profile_dir / "baseline_bench.json"),
        "benchmark": bench,
        "decode_profile": profile,
    }
    (output_dir / f"m7_candidate_{commit}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
