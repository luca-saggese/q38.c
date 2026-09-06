#!/usr/bin/env python3
"""Run and normalize the single canonical Q2 benchmark path."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SCHEMA = "PERF_SCHEMA_V2"
DEFAULT_MODEL = (
    "artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf"
)
DEFAULT_TOKENIZER = "/home/lvx/q38model"
DEFAULT_PROMPT = (
    "Explain in simple terms why the sky appears blue during the day "
    "and red near sunset."
)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def model_fingerprint(path: Path) -> dict[str, object]:
    checksums = Path("artifacts/m1/checksums.txt")
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and Path(parts[1]).as_posix() == path.as_posix():
                return {"kind": "sha256", "value": parts[0]}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(1024 * 1024))
        if path.stat().st_size > 2 * 1024 * 1024:
            stream.seek(-1024 * 1024, os.SEEK_END)
            digest.update(stream.read(1024 * 1024))
    return {
        "kind": "sampled_sha256",
        "value": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def quant_recipe(model: Path) -> dict[str, object]:
    manifest = (
        Path("artifacts/m8/quant_manifest_R1.json")
        if "R1-" in model.name
        else Path("tools/quant_manifest_q2.json")
    )
    if not manifest.exists():
        return {"manifest": str(manifest), "status": "unavailable"}
    return {"manifest": str(manifest), "sha256": hashlib.sha256(
        manifest.read_bytes()).hexdigest()}


def run_raw(args: argparse.Namespace) -> dict[str, object]:
    command = [
        args.binary,
        "--mode",
        args.mode,
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--prompt",
        args.prompt,
        "--ctx",
        str(args.ctx),
        "--prefill-chunk",
        str(args.prefill_chunk),
    ]
    if args.mode == "decode":
        command += [
            "--generated",
            str(args.generated),
            "--measure-first",
            str(args.measure_first),
            "--measure-last",
            str(args.measure_last),
        ]
    else:
        command += ["--prefill-sizes", args.prefill_sizes]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        sys.stderr.write(completed.stdout)
        raise SystemExit(f"canonical benchmark emitted invalid JSON: {exc}")


def normalize(raw: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    model = Path(args.model)
    binary = Path(args.binary)
    commit_hash = git("rev-parse", "HEAD")
    raw["schema_version"] = SCHEMA
    raw["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    raw["build"] = {
        "git_commit": commit_hash,
        "git_dirty": bool(git("status", "--porcelain")),
        "binary": str(binary),
        "binary_size_bytes": binary.stat().st_size if binary.exists() else None,
        "cflags": os.environ.get("CFLAGS", "Makefile default"),
        "nvccflags": os.environ.get("NVCCFLAGS", "Makefile default"),
    }
    raw["model"] = {
        "path": str(model),
        "file_size_bytes": model.stat().st_size,
        "fingerprint": model_fingerprint(model),
        "quant_recipe": quant_recipe(model),
    }
    raw["execution"] = {
        "path": (
            "q38_runtime -> q38_session -> "
            "q38_session_prefill -> q38_session_eval"
        ),
        "backend_configuration": "installed by q38_session runtime",
        "sampling": "greedy",
        "ple_policy": "file_backed",
        "non_ple_residency": "resident_required",
        "timed_region": "session eval/prefill wall; no state trace",
    }
    raw["timer_schema"] = {
        "version": SCHEMA,
        "span_policy": {
            "wall": "exclusive measured wall",
            "stage": "exclusive stage timer",
            "cuda_dispatch": "diagnostic_span_nonadditive",
            "cuda_sync_wait": "diagnostic_span_nonadditive",
            "memcpy": "diagnostic_span_nonadditive",
            "PLE_elapsed": "diagnostic_span_nonadditive",
            "PLE_overlap": "diagnostic_span_nonadditive",
            "PLE_critical_stall": "exclusive critical-path contribution",
        },
        "categories": [
            "wall",
            "QSA",
            "QSA.qkv",
            "QSA.output_projection",
            "QSA.attention",
            "QSA.index_compress",
            "QSA.state_glue",
            "MoE",
            "GDN",
            "GR",
            "norms_residual_glue",
            "LM_head",
            "argmax",
            "host_scalar",
            "cuda_dispatch",
            "cuda_sync_wait",
            "memcpy",
            "PLE_critical_stall",
            "other",
        ],
    }
    raw["release_gating"] = {
        "canonical": True,
        "legacy_benchmarks_release_gating": False,
    }
    # Keep the comparison gates explicit at the top level as well as in the
    # structured metadata, so legacy artifacts cannot be mistaken for V2.
    raw["commit_hash"] = commit_hash
    raw["execution_path"] = raw["execution"]["path"]
    raw["timer_schema_version"] = SCHEMA
    raw["model_fingerprint"] = raw["model"]["fingerprint"]
    raw["quant_recipe"] = raw["model"]["quant_recipe"]
    raw["ple_policy"] = raw["execution"]["ple_policy"]
    if isinstance(raw.get("residency"), dict):
        raw["resident_bytes"] = raw["residency"].get("persistent_resident_bytes")
    if args.mode == "decode":
        raw["position"] = {
            "prefill_final": len(raw.get("prompt_ids", [])),
            "measured_first": args.measure_first,
            "measured_last": args.measure_last,
            "context_size": args.ctx,
        }
    else:
        raw["position"] = {
            "prefill_sizes": args.prefill_sizes,
            "context_size": args.ctx,
        }
    return raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("decode", "prefill"), required=True)
    parser.add_argument("--binary", default="./tests/q2_canonical_bench")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument("--generated", type=int, default=128)
    parser.add_argument("--measure-first", type=int, default=16)
    parser.add_argument("--measure-last", type=int, default=127)
    parser.add_argument("--prefill-sizes", default="128,512,2048")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = normalize(run_raw(args), args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
