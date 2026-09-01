#!/usr/bin/env python3
"""Generate the M4-C06 file-backed PLE reference corpus.

This program deliberately does not import or execute q38 code.  It reads the
HF configuration and the three small int64 PLE metadata tensors directly from
the safetensors headers/data, then applies the hash loop from the frozen
llama.cpp reference.  The 336 GiB checkpoint is not loaded: the current
runtime has no complete model-forward implementation, so hidden-state fields
are explicitly emitted as null rather than fabricated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


SOURCE_REVISION = "ggml-org/llama.cpp@refs/pull/27742/head"
SOURCE_FILES = {
    "graph": "src/models/qwen4exp.cpp",
    "converter": "conversion/qwen4exp.py",
}
SOURCE_SHA256 = {
    "src/models/qwen4exp.cpp": "c293b3408f3c73283457996f4cc5ca65c26e27e2c5097db03221db73b912e518",
    "conversion/qwen4exp.py": "12a0a5aea7877fbb8fe35af041a9c34f8b57b05278871b22c24c650b9760dfc3",
}


def safetensors_tensor(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as fp:
        header_len = struct.unpack("<Q", fp.read(8))[0]
        header = json.loads(fp.read(header_len))
        info = header[name]
        start, end = info["data_offsets"]
        fp.seek(8 + header_len + start)
        return info, fp.read(end - start)


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as fp:
        header_len = struct.unpack("<Q", fp.read(8))[0]
        return json.loads(fp.read(header_len))


def read_i64(index: dict[str, Any], model_dir: Path, name: str) -> list[int]:
    shard = model_dir / index["weight_map"][name]
    info, payload = safetensors_tensor(shard, name)
    if info["dtype"] != "I64" or info["shape"] != [len(payload) // 8]:
        raise ValueError(f"{name}: expected a one-dimensional I64 tensor")
    return list(struct.unpack("<" + "q" * (len(payload) // 8), payload))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_case(
    name: str, tokens: list[int], cfg: dict[str, Any],
    partitions: list[int] | None = None,
) -> dict[str, Any]:
    eos = cfg["eos_token_id"]
    multipliers = cfg["multipliers"]
    offsets = cfg["head_offsets"]
    vocab_sizes = cfg["head_vocab_sizes"]
    rows: list[list[int]] = []
    previous: list[int] = []
    for token in tokens:
        context = [token]
        cut = False
        for distance in (1, 2):
            predecessor = previous[-distance] if len(previous) >= distance else eos
            cut = cut or predecessor == eos
            context.append(eos if cut else predecessor)
        token_rows: list[int] = []
        for order in (2, 3):
            mixed = (context[0] * multipliers[0]) & ((1 << 64) - 1)
            for j in range(1, order):
                mixed ^= (context[j] * multipliers[j]) & ((1 << 64) - 1)
            base = (order - 2) * cfg["heads_per_ngram"]
            for head in range(cfg["heads_per_ngram"]):
                h = base + head
                token_rows.append(mixed % vocab_sizes[h] + offsets[h])
        rows.append(token_rows)
        previous.append(token)
        if token == eos:
            previous = [eos]
    return {
        "name": name,
        "tokens": tokens,
        "partitions": partitions or [len(tokens)],
        "ngram_row_ids": rows,
        "hidden_before_ple": None,
        "hidden_after_ple": None,
        "ple_contribution_vector": None,
    }


def generate(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config_file = json.loads(config_path.read_text())
    text = config_file["text_config"]
    index = json.loads(index_path.read_text())

    names = {
        "multipliers": "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        "head_offsets": "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
        "head_vocab_sizes": "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
    }
    constants = {
        key: read_i64(index, model_dir, name) for key, name in names.items()
    }
    shard_prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
    shard_names = [
        name for name in index["weight_map"]
        if name.startswith(shard_prefix) and name.endswith(".weight")
    ]
    expected_shards = int(text["split_ngram_parts"])
    if len(shard_names) != expected_shards:
        raise ValueError(f"expected {expected_shards} PLE shards, found {len(shard_names)}")
    shard_infos: list[dict[str, Any]] = []
    for shard_index in range(expected_shards):
        name = f"{shard_prefix}{shard_index}.weight"
        shard_path = model_dir / index["weight_map"][name]
        info = safetensors_header(shard_path).get(name)
        if info is None:
            raise ValueError(f"missing safetensors header for {name}")
        shard_infos.append(info)
    shard_info = shard_infos[0]
    row_count, row_width = shard_info["shape"]
    if shard_info["dtype"] != "BF16" or row_width <= 0:
        raise ValueError("unexpected PLE shard dtype or shape")
    if any(info["dtype"] != shard_info["dtype"] or
           info["shape"][1] != row_width for info in shard_infos):
        raise ValueError("PLE shards do not have a consistent BF16 row width")
    total_rows = sum(int(info["shape"][0]) for info in shard_infos)

    cfg = {
        "ngram_size": int(text["ngram_size"]),
        "heads_per_ngram": int(text["heads_per_ngram"]),
        "eos_token_id": int(text["eos_token_id"]),
        **constants,
    }
    if cfg["ngram_size"] != 3 or cfg["heads_per_ngram"] != 8:
        raise ValueError("M4-C06 corpus expects the frozen 3-gram/8-head model")
    if any(len(constants[key]) != (3 if key == "multipliers" else 16)
           for key in constants):
        raise ValueError("unexpected PLE constant lengths")

    vocab = int(text["vocab_size"])
    bos = int(text["bos_token_id"])
    eos = cfg["eos_token_id"]
    cases = [
        make_case("bos_only", [bos], cfg),
        make_case("boundary_tokens", [0, 1, vocab - 1], cfg),
        make_case("repeated_token", [7, 7, 7, 7, 7], cfg),
        make_case("eos_reset", [11, eos, 22, 33], cfg),
        make_case("chunk_a_bc", [101, 102, 103], cfg, [1, 2]),
        make_case("chunk_ab_c", [401, 402, 403], cfg, [2, 1]),
        make_case("one_token_chunks", list(range(17, 29)), cfg, [1] * 12),
    ]
    corpus_bytes = json.dumps(
        [{"name": c["name"], "tokens": c["tokens"],
          "ngram_row_ids": c["ngram_row_ids"]} for c in cases],
        sort_keys=True, separators=(",", ":"),
    ).encode()

    return {
        "format": "q38-m4-c06-ple-golden-v1",
        "source": {
            "revision": SOURCE_REVISION,
            "files": SOURCE_FILES,
            "sha256": SOURCE_SHA256,
        },
        "model": {
            "revision": f"Qwen3.8-Flash-Next / Transformers {config_file.get('transformers_version', 'unknown')}",
            "config_sha256": sha256_file(config_path),
            "index_sha256": sha256_file(index_path),
            "checkpoint_layout": {
                "shard_count": int(text["split_ngram_parts"]),
                "shard_rows": int(row_count),
                "total_rows": int(total_rows),
                "physical_row_width": int(row_width),
                "logical_embedding_width": int(row_width) * 16,
                "dtype": shard_info["dtype"],
            },
        },
        "config": {
            "ngram_size": cfg["ngram_size"],
            "heads_per_ngram": cfg["heads_per_ngram"],
            "eos_token_id": eos,
            "multipliers": constants["multipliers"],
            "head_offsets": constants["head_offsets"],
            "head_vocab_sizes": constants["head_vocab_sizes"],
        },
        "coverage": {
            "token_ids": "verified against checkpoint int64 constants",
            "ngram_row_ids": "verified against frozen llama.cpp host hash loop",
            "ple_injection_boundary": "hidden + gated + conv_out",
            "full_model_forward": "unavailable",
            "hidden_vectors": "not present; no complete q38 forward graph or affordable full-checkpoint execution",
        },
        "checksums": {
            "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/home/lvx/q38model"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/m4/ple_injection_golden.json"))
    args = parser.parse_args()
    output = generate(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
