#!/usr/bin/env python3
"""Run and normalize the single canonical Q2 benchmark path."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
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
DEFAULT_CFLAGS = (
    "-O3 -g -Wall -Wextra -std=c99 -D_GNU_SOURCE "
    "-fno-finite-math-only -I. -pthread"
)
DEFAULT_NVCCFLAGS = (
    "-O3 -g -lineinfo --use_fast_math "
    "-gencode arch=compute_121a,code=sm_121a"
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


def tokenizer_fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    files = []
    names = {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    selected = [
        p for p in path.iterdir()
        if p.is_file() and p.name in names
    ]
    for child in sorted(selected):
        relative = child.relative_to(path).as_posix()
        files.append(relative)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
    return {"kind": "sha256", "value": digest.hexdigest(), "files": files}


def command_version(command: str) -> str:
    try:
        return subprocess.check_output(
            [command, "--version"], text=True, stderr=subprocess.STDOUT
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unavailable"


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int((len(ordered) * percentile_value) + 0.999999))
    return ordered[min(rank, len(ordered)) - 1]


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "median": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
            "stddev": 0.0,
        }
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def sample_breakdown(samples: list[dict[str, object]]) -> dict[str, object]:
    def values(path: tuple[str, ...]) -> list[float]:
        result = []
        for sample in samples:
            value: object = sample
            for key in path:
                if not isinstance(value, dict):
                    value = 0.0
                    break
                value = value.get(key, 0.0)
            if isinstance(value, (int, float)):
                result.append(float(value))
        return result

    categories = {
        "QSA": ("categories", "QSA", "ms"),
        "MoE": ("categories", "MoE", "ms"),
        "GDN": ("categories", "GDN", "ms"),
        "GR": ("categories", "GR", "ms"),
        "norms_residual_glue": ("categories", "norms_residual_glue", "ms"),
        "LM_head": ("categories", "LM_head", "ms"),
        "argmax": ("argmax_ms",),
        "host_scalar": ("categories", "host_scalar", "ms"),
        "cuda_sync_wait": ("categories", "cuda_sync_wait", "ms"),
        "cuda_dispatch": ("categories", "cuda_dispatch", "ms"),
        "memcpy": ("categories", "memcpy", "ms"),
        "PLE_critical_stall": ("categories", "PLE_critical_stall", "ms"),
        "other": ("categories", "other", "ms"),
        "PLE_elapsed": ("ple_elapsed_ms",),
        "PLE_overlap": ("ple_overlap_ms",),
    }
    result = {
        "wall_ms": distribution(values(("wall_ms",))),
        "forward_core_ms": distribution(values(("forward_core_ms",))),
        "tok_s": {
            "median": 1000.0 / distribution(values(("wall_ms",)))["median"]
            if values(("wall_ms",)) else 0.0
        },
        "categories": {
            name: distribution(values(path)) for name, path in categories.items()
        },
        "traffic_per_token": {
            name: statistics.mean(values(("traffic", name)))
            if values(("traffic", name)) else 0.0
            for name in (
                "kernel_launches",
                "host_syncs",
                "h2d_bytes",
                "d2h_bytes",
                "d2d_bytes",
            )
        },
        "sample_count": len(samples),
    }
    result["nonadditive"] = {
        "PLE_elapsed_ms": result["categories"]["PLE_elapsed"],
        "PLE_overlap_ms": result["categories"]["PLE_overlap"],
    }
    return result


def run_raw(args: argparse.Namespace) -> dict[str, object]:
    raw_mode = "reference0" if args.mode.startswith("current-") else args.mode
    command = [
        args.binary,
        "--mode",
        raw_mode,
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
    if raw_mode == "decode":
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


def common_metadata(raw: dict[str, object],
                    args: argparse.Namespace) -> dict[str, object]:
    model = Path(args.model)
    binary = Path(args.binary)
    commit_hash = git("rev-parse", "HEAD")
    build = {
        "git_commit": commit_hash,
        "git_dirty": bool(git("status", "--porcelain")),
        "branch": git("branch", "--show-current"),
        "binary": str(binary),
        "binary_size_bytes": binary.stat().st_size if binary.exists() else None,
        "compiler": command_version(os.environ.get("CC", "cc")),
        "nvcc": command_version(os.environ.get("NVCC", "nvcc")),
        "cflags": os.environ.get("CFLAGS", DEFAULT_CFLAGS),
        "nvccflags": os.environ.get("NVCCFLAGS", DEFAULT_NVCCFLAGS),
        "cuda_arch": (
            re.search(r"sm_[0-9]+[a-z]?", os.environ.get(
                "NVCCFLAGS", DEFAULT_NVCCFLAGS)).group(0)
            if re.search(r"sm_[0-9]+[a-z]?", os.environ.get(
                "NVCCFLAGS", DEFAULT_NVCCFLAGS)) else "sm_121"
        ),
        "build_type": "production -O3",
    }
    model_metadata = {
        "path": str(model),
        "file_size_bytes": model.stat().st_size,
        "fingerprint": model_fingerprint(model),
        "quant_recipe": quant_recipe(model),
    }
    execution = {
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
    timer_schema = {
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
    return {
        "schema_version": SCHEMA,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "build": build,
        "model": model_metadata,
        "tokenizer": {
            "path": str(args.tokenizer),
            "fingerprint": tokenizer_fingerprint(Path(args.tokenizer)),
        },
        "hardware": raw.get("hardware", {}),
        "execution": execution,
        "timer_schema": timer_schema,
        "release_gating": {
            "canonical": True,
            "legacy_benchmarks_release_gating": False,
        },
        "commit_hash": commit_hash,
        "execution_path": execution["path"],
        "timer_schema_version": SCHEMA,
        "model_fingerprint": model_metadata["fingerprint"],
        "quant_recipe": model_metadata["quant_recipe"],
        "ple_policy": execution["ple_policy"],
    }


def normalize_reference0(raw: dict[str, object],
                         args: argparse.Namespace) -> tuple[dict[str, object],
                                                            dict[str, object]]:
    metadata = common_metadata(raw, args)
    decode_raw = raw["decode"]
    prefill_raw = raw["prefill"]
    decode_runs = decode_raw["runs"]
    samples = [
        sample
        for run in decode_runs
        for sample in run["samples"]
    ]
    decode = dict(metadata)
    decode.update({
        "ref_id": "Q2_DECODE_REFERENCE_0",
        "REF_ID": "Q2_DECODE_REFERENCE_0",
        "mode": "decode",
        "prompt": raw["prompt"],
        "prompt_ids": raw["prompt_ids"],
        "generated_ids": decode_runs[0]["generated_ids"],
        "measured_positions": list(range(
            int(raw["measure_first"]), int(raw["measure_last"]) + 1)),
        "context_size": raw["context_size"],
        "generated_count": raw["generated_count"],
        "warmup_runs": raw["warmup_runs"],
        "measured_runs": raw["measured_runs"],
        "prefill_chunk": raw["prefill_chunk"],
        "run_summaries": [run["summary"] for run in decode_runs],
        "metrics": sample_breakdown(samples),
        "correctness": {
            "green": raw["correctness"]["decode"] and
            raw["correctness"]["all_finite"],
            "final_logits_hash": decode_runs[0]["correctness"]["final_logits_hash"],
            "run_hashes": [
                run["correctness"]["final_logits_hash"] for run in decode_runs
            ],
            "generated_ids_exact": all(
                run["generated_ids"] == decode_runs[0]["generated_ids"]
                for run in decode_runs
            ),
            "nan_inf": not raw["correctness"]["all_finite"],
        },
        "residency": raw["residency"],
        "memory": raw["memory"],
        "hardware": raw["hardware"],
        "comparison_key": {
            "model_fingerprint": metadata["model_fingerprint"],
            "quant_recipe": metadata["quant_recipe"],
            "tokenizer_fingerprint": metadata["tokenizer"]["fingerprint"],
            "execution_path": metadata["execution_path"],
            "input_token_ids": raw["prompt_ids"],
            "context_reset": "semantic RESET between runs",
            "residency_policy": metadata["execution"]["non_ple_residency"],
            "build_mode": metadata["build"]["build_type"],
            "timer_schema_version": SCHEMA,
            "measurement_window": [raw["measure_first"], raw["measure_last"]],
        },
    })
    prefill = dict(metadata)
    prefill.update({
        "ref_id": "Q2_PREFILL_REFERENCE_0",
        "REF_ID": "Q2_PREFILL_REFERENCE_0",
        "mode": "prefill",
        "prompt": raw["prompt"],
        "prompt_ids": raw["prompt_ids"],
        "context_size": raw["context_size"],
        "prefill_chunk": raw["prefill_chunk"],
        "warmup_runs": raw["warmup_runs"],
        "measured_runs": raw["measured_runs"],
        "workloads": [],
        "residency": raw["residency"],
        "memory": raw["memory"],
        "hardware": raw["hardware"],
    })
    for case in prefill_raw["cases"]:
        case_samples = [run["sample"] for run in case["measured_runs"]]
        workload = {
            "token_count": case["token_count"],
            "tokens": case["tokens"],
            "metrics": sample_breakdown(case_samples),
            "ttft_ms": distribution([
                sample["wall_ms"] for sample in case_samples
            ]),
            "next_token": case["measured_runs"][0]["next_token"],
            "logits_hash": case["measured_runs"][0]["logits_hash"],
            "correctness": {
                "green": raw["correctness"]["prefill"] and
                all(run["nan_inf"] is False
                    for run in case["measured_runs"]),
                "state_equivalent": all(
                    run["logits_hash"] == case["measured_runs"][0]["logits_hash"]
                    and run["next_token"] ==
                    case["measured_runs"][0]["next_token"]
                    for run in case["measured_runs"]
                ),
            },
        }
        prefill["workloads"].append(workload)
    prefill["comparison_key"] = {
        "model_fingerprint": metadata["model_fingerprint"],
        "quant_recipe": metadata["quant_recipe"],
        "tokenizer_fingerprint": metadata["tokenizer"]["fingerprint"],
        "execution_path": metadata["execution_path"],
        "context_reset": "semantic RESET between runs",
        "residency_policy": metadata["execution"]["non_ple_residency"],
        "build_mode": metadata["build"]["build_type"],
        "timer_schema_version": SCHEMA,
        "measurement_window": [128, 512, 2048],
    }
    return decode, prefill


def normalize(raw: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    metadata = common_metadata(raw, args)
    raw.update(metadata)
    raw["release_gating"] = {
        "canonical": True,
        "legacy_benchmarks_release_gating": False,
    }
    # Keep the comparison gates explicit at the top level as well as in the
    # structured metadata, so legacy artifacts cannot be mistaken for V2.
    if isinstance(raw.get("residency"), dict):
        raw["resident_bytes"] = raw["residency"].get("persistent_resident_bytes")
    if args.mode in ("decode", "reference0"):
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


def normalize_current(raw: dict[str, object],
                      args: argparse.Namespace) -> dict[str, object]:
    decode, prefill = normalize_reference0(raw, args)
    if args.mode == "current-decode":
        decode.pop("ref_id", None)
        decode.pop("REF_ID", None)
        decode["artifact_id"] = "Q2_DECODE_CANONICAL_CURRENT"
        decode["immutable"] = False
        return decode
    prefill.pop("ref_id", None)
    prefill.pop("REF_ID", None)
    prefill["artifact_id"] = "Q2_PREFILL_CANONICAL_CURRENT"
    prefill["immutable"] = False
    return prefill


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("decode", "prefill", "reference0",
                 "current-decode", "current-prefill"),
        required=True,
    )
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
    parser.add_argument("--output")
    parser.add_argument("--output-dir", default="artifacts/perf/reference_0")
    args = parser.parse_args()

    if args.mode == "reference0":
        output_dir = Path(args.output_dir)
        decode_output = output_dir / "q2_decode_reference_0.json"
        prefill_output = output_dir / "q2_prefill_reference_0.json"
        if decode_output.exists() or prefill_output.exists():
            raise SystemExit("Reference 0 is immutable and already exists")
        decode, prefill = normalize_reference0(run_raw(args), args)
        output_dir.mkdir(parents=True, exist_ok=True)
        decode_output.write_text(
            json.dumps(decode, indent=2, sort_keys=True) + "\n"
        )
        prefill_output.write_text(
            json.dumps(prefill, indent=2, sort_keys=True) + "\n"
        )
    elif args.mode.startswith("current-"):
        if not args.output:
            raise SystemExit("--output is required for current benchmark mode")
        result = normalize_current(run_raw(args), args)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        if not args.output:
            raise SystemExit("--output is required for decode/prefill mode")
        result = normalize(run_raw(args), args)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
