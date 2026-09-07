#!/usr/bin/env python3
"""Extract only the production Q8_0 PLE projection/row slices from GGUF.

This maps and parses GGUF metadata directly. It never constructs a runtime
model or loads any non-PEL tensor.
"""

import argparse
import hashlib
import json
import mmap
import struct
from pathlib import Path

import numpy as np


def string(buf, pos):
    length = struct.unpack_from("<Q", buf, pos)[0]
    pos += 8
    return bytes(buf[pos : pos + length]).decode("utf-8"), pos + length


def skip_value(buf, pos, kind):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if kind in sizes:
        return pos + sizes[kind]
    if kind == 8:
        length = struct.unpack_from("<Q", buf, pos)[0]
        return pos + 8 + length
    if kind == 9:
        item_kind = struct.unpack_from("<I", buf, pos)[0]
        count = struct.unpack_from("<Q", buf, pos + 4)[0]
        pos += 12
        for _ in range(count):
            pos = skip_value(buf, pos, item_kind)
        return pos
    raise ValueError(f"unsupported GGUF metadata type {kind}")


def parse_tensors(path):
    with path.open("rb") as file:
        size = path.stat().st_size
        mapped = mmap.mmap(file.fileno(), size, access=mmap.ACCESS_READ)
        magic, version = struct.unpack_from("<II", mapped, 0)
        if magic != 0x46554747 or version != 3:
            raise ValueError("not GGUF v3")
        tensor_count, kv_count = struct.unpack_from("<QQ", mapped, 8)
        pos = 24
        for _ in range(kv_count):
            _, pos = string(mapped, pos)
            kind = struct.unpack_from("<I", mapped, pos)[0]
            pos = skip_value(mapped, pos + 4, kind)
        descriptors = []
        for _ in range(tensor_count):
            name, pos = string(mapped, pos)
            ndim = struct.unpack_from("<I", mapped, pos)[0]
            pos += 4
            dims = struct.unpack_from("<" + "Q" * ndim, mapped, pos)
            pos += 8 * ndim
            qtype = struct.unpack_from("<I", mapped, pos)[0]
            rel_offset = struct.unpack_from("<Q", mapped, pos + 4)[0]
            pos += 12
            descriptors.append((name, dims, qtype, rel_offset))
        alignment = 32
        data_pos = (pos + alignment - 1) & ~(alignment - 1)
        return mapped, data_pos, descriptors


def q8_row(raw, row_bytes, row):
    offset = row * row_bytes
    result = np.empty(2560, dtype=np.float32)
    for block in range(80):
        start = offset + block * 34
        scale = np.frombuffer(raw[start : start + 2], dtype="<f2")[0].astype(
            np.float32
        )
        values = np.frombuffer(raw[start + 2 : start + 34], dtype=np.int8)
        result[block * 32 : block * 32 + 32] = scale * values
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument(
        "--vectors",
        type=Path,
        default=Path("artifacts/m4/ple_injection_vectors.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/m4/ple_projection_q8_fixture")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mapped, data_pos, descriptors = parse_tensors(args.gguf)
    wanted = {
        "model.language_model.layers.1.ple.key_proj.weight",
        "model.language_model.layers.1.ple.value_proj.weight",
    }
    projections = {}
    shards = {}
    for name, dims, qtype, rel in descriptors:
        if name in wanted:
            projections[name] = (dims, qtype, rel)
        if ".ple.ple_embedding.ngram_embedding.shard_" in name:
            shards[name] = (dims, qtype, rel)
    if len(projections) != 2:
        raise ValueError(f"missing projection tensors: {sorted(projections)}")
    for name, (dims, qtype, _) in projections.items():
        if tuple(dims) not in ((10240, 2560), (2560, 2560)) or qtype != 8:
            raise ValueError(f"unexpected projection contract for {name}: {dims}/{qtype}")
    vectors = json.loads(args.vectors.read_text())
    selected = [0, 1, 3]
    row_ids = [vectors["row_ids"][index] for index in selected]
    rows = np.empty((3, 16, 160), dtype=np.float32)
    row_bytes = 170
    for case, ids in enumerate(row_ids):
        for head, global_row in enumerate(ids):
            shard = global_row // 2500012
            local = global_row % 2500012
            name = (
                "model.language_model.layers.1.ple.ple_embedding."
                f"ngram_embedding.shard_{shard}.weight"
            )
            dims, qtype, rel = shards[name]
            if tuple(dims) != (2500012, 160) or qtype != 8:
                raise ValueError(f"unexpected PLE shard contract for {name}")
            start = data_pos + rel + local * row_bytes
            decoded = np.empty(160, dtype=np.float32)
            for block in range(5):
                block_start = start + block * 34
                scale = np.frombuffer(mapped[block_start : block_start + 2], dtype="<f2")[
                    0
                ].astype(np.float32)
                values = np.frombuffer(
                    mapped[block_start + 2 : block_start + 34], dtype=np.int8
                )
                decoded[block * 32 : block * 32 + 32] = scale * values
            rows[case, head] = decoded

    weights = {}
    for name, (dims, qtype, rel) in projections.items():
        elements = int(np.prod(dims))
        raw = bytes(mapped[data_pos + rel : data_pos + rel + elements // 32 * 34])
        weights[name] = np.frombuffer(raw, dtype=np.uint8).copy()
    embedding = rows.reshape(3, 2560)

    def project(raw, shape):
        matrix = np.empty(shape, dtype=np.float32)
        row_bytes = shape[1] // 32 * 34
        for row in range(shape[0]):
            matrix[row] = q8_row(raw, row_bytes, row)
        return embedding @ matrix.T

    key_name = "model.language_model.layers.1.ple.key_proj.weight"
    value_name = "model.language_model.layers.1.ple.value_proj.weight"
    key_expected = project(weights[key_name], (10240, 2560))
    value_expected = project(weights[value_name], (2560, 2560))
    hidden = np.asarray(
        [vectors["hidden_before_ple"][index] for index in selected], dtype=np.float32
    )
    del hidden
    for name, array in (
        ("rows_q8_f32.bin", rows),
        ("key_proj_q8.bin", weights[key_name]),
        ("value_proj_q8.bin", weights[value_name]),
        ("key_expected_f32.bin", key_expected),
        ("value_expected_f32.bin", value_expected),
    ):
        array.tofile(args.output / name)
    metadata = {
        "format": "q38-ple-projection-q8-fixture-v1",
        "oracle": "PLE_PROJ_Q8_ORACLE_V1",
        "cases": ["early", "middle", "late"],
        "tokens": [vectors["tokens"][index] for index in selected],
        "row_ids": row_ids,
        "source_gguf": str(args.gguf),
        "source_weight_type": "Q8_0",
        "row_type": "Q8_0",
        "row_width": 160,
        "embedding_width": 2560,
        "key_shape": [10240, 2560],
        "value_shape": [2560, 2560],
        "key_row_bytes": 2720,
        "value_row_bytes": 2720,
        "files": {},
    }
    for file in sorted(args.output.iterdir()):
        metadata["files"][file.name] = {
            "bytes": file.stat().st_size,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
    (args.output / "fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    mapped.close()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
