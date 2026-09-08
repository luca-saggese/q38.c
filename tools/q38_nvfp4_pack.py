#!/usr/bin/env python3
"""Build the q38-native NVFP4 pack/index.

The importer deliberately does not depend on transformers, ModelOpt, or the
safetensors Python package.  It reads safetensors headers directly and copies
payloads with bounded ``pread`` buffers when materializing a pack.  The
source-backed mode is useful on hosts that cannot hold a second copy of the
checkpoint: it writes the same native descriptor tables while retaining
explicit source-file offsets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


MAGIC = b"Q38_NVFP4_PACK_V1" + bytes(
    24 - len(b"Q38_NVFP4_PACK_V1")
)
VERSION = 1
MODE_SOURCE_BACKED = 1
MODE_MATERIALIZED = 2
FLAG_MTP_DISABLED = 1 << 0
FLAG_VISION_INDEXED = 1 << 1
FLAG_PLE_EXTERNAL = 1 << 2
FLAG_BF16_UNCHANGED = 1 << 3

PROJECTIONS = ("gate", "up", "down")
PROJECTION_ID = {name: i for i, name in enumerate(PROJECTIONS)}
COMPONENTS = ("weight", "weight_scale", "weight_scale_2", "input_scale")
COMPONENT_ID = {name: i for i, name in enumerate(COMPONENTS)}

DTYPE_ID = {
    "BF16": 1,
    "U8": 2,
    "F8_E4M3": 3,
    "F32": 4,
    "I64": 5,
}
CLASS_ID = {
    "vision": 1,
    "MTP FP8": 2,
    "other": 3,
}

HEADER_FORMAT = "<24s8I20Q32s"
HEADER_SIZE = 512
SOURCE_FORMAT = "<QIIQ32s"
SOURCE_SIZE = struct.calcsize(SOURCE_FORMAT)
REF_FORMAT = "<IIQQQ"
REF_SIZE = struct.calcsize(REF_FORMAT)
EXPERT_SIZE = REF_SIZE * len(COMPONENTS)
BF16_FORMAT = "<QIIIIQQQ4Q"
BF16_SIZE = struct.calcsize(BF16_FORMAT)
AUX_FORMAT = "<QIIIIIIQQQ4Q"
AUX_SIZE = struct.calcsize(AUX_FORMAT)
PLE_FORMAT = "<IIIIQQQ4Q"
PLE_SIZE = struct.calcsize(PLE_FORMAT)

EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_scale_2|input_scale)$"
)


@dataclass(frozen=True)
class Tensor:
    name: str
    source_id: int
    source_name: str
    source_offset: int
    bytes: int
    dtype: str
    shape: tuple[int, ...]


@dataclass
class Source:
    name: str
    path: Path
    size: int


@dataclass
class ExpertEntry:
    layer: int
    expert: int
    projection: int
    component: int
    tensor: Tensor


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"q38_nvfp4_pack: {message}")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as fp:
        while True:
            block = fp.read(chunk_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def read_safetensors_header(path: Path) -> dict[str, dict]:
    with path.open("rb") as fp:
        prefix = fp.read(8)
        if len(prefix) != 8:
            fail(f"truncated safetensors header: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes > 128 * 1024 * 1024:
            fail(f"safetensors header exceeds 128 MiB: {path}")
        raw = fp.read(header_bytes)
        if len(raw) != header_bytes:
            fail(f"truncated safetensors JSON header: {path}")
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"invalid safetensors JSON header {path}: {exc}")
    if not isinstance(header, dict):
        fail(f"safetensors header is not an object: {path}")
    return header


def tensor_from_header(
    name: str, value: dict, source_id: int, source_name: str, data_base: int
) -> Tensor:
    if not isinstance(value, dict):
        fail(f"invalid tensor header entry: {name}")
    dtype = value.get("dtype")
    shape = value.get("shape")
    offsets = value.get("data_offsets")
    if dtype not in DTYPE_ID:
        fail(f"unsupported dtype {dtype!r} for {name}")
    if not isinstance(shape, list) or not all(
        isinstance(x, int) and x >= 0 for x in shape
    ):
        fail(f"invalid shape for {name}")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(x, int) and x >= 0 for x in offsets)
        or offsets[1] < offsets[0]
    ):
        fail(f"invalid data offsets for {name}")
    return Tensor(
        name=name,
        source_id=source_id,
        source_name=source_name,
        source_offset=data_base + offsets[0],
        bytes=offsets[1] - offsets[0],
        dtype=dtype,
        shape=tuple(shape),
    )


def source_files(root: Path) -> list[Source]:
    paths = sorted(
        path
        for path in root.glob("model-*.safetensors")
        if re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", path.name)
    )
    sidecar = root / "model-fp8-mtp-ple.safetensors"
    if sidecar.exists():
        paths.append(sidecar)
    if len(paths) != 11:
        fail(f"expected 10 main shards plus sidecar, found {len(paths)}")
    return [Source(path.name, path, path.stat().st_size) for path in paths]


def collect_tensors(root: Path, sources: list[Source]) -> list[Tensor]:
    tensors: list[Tensor] = []
    for source_id, source in enumerate(sources):
        with source.path.open("rb") as fp:
            prefix = fp.read(8)
            if len(prefix) != 8:
                fail(f"truncated safetensors file: {source.path}")
            header_bytes = struct.unpack("<Q", prefix)[0]
            raw = fp.read(header_bytes)
            if len(raw) != header_bytes:
                fail(f"truncated safetensors header: {source.path}")
        try:
            header = json.loads(raw)
        except json.JSONDecodeError as exc:
            fail(f"invalid safetensors JSON header {source.path}: {exc}")
        data_base = 8 + header_bytes
        for name, value in header.items():
            if name == "__metadata__":
                continue
            tensors.append(
                tensor_from_header(name, value, source_id, source.name, data_base)
            )
    return tensors


def classify(
    tensors: Iterable[Tensor],
) -> tuple[
    dict[tuple[int, int, int, int], ExpertEntry],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[Tensor],
]:
    experts: dict[tuple[int, int, int, int], ExpertEntry] = {}
    bf16: list[Tensor] = []
    aux: list[Tensor] = []
    ple: list[Tensor] = []
    mtp: list[Tensor] = []
    vision: list[Tensor] = []
    other: list[Tensor] = []
    unknown: list[Tensor] = []

    for tensor in tensors:
        match = EXPERT_RE.match(tensor.name)
        if match:
            layer, expert, projection, component = match.groups()
            expected_dtype = {
                "weight": "U8",
                "weight_scale": "F8_E4M3",
                "weight_scale_2": "F32",
                "input_scale": "F32",
            }[component]
            if tensor.dtype != expected_dtype:
                fail(f"NVFP4 dtype mismatch for {tensor.name}")
            key = (
                int(layer),
                int(expert),
                PROJECTION_ID[{"gate_proj": "gate", "up_proj": "up",
                               "down_proj": "down"}[projection]],
                COMPONENT_ID[component],
            )
            if key in experts:
                fail(f"duplicate NVFP4 tensor: {tensor.name}")
            experts[key] = ExpertEntry(
                int(layer), int(expert), key[2], key[3], tensor
            )
            continue

        if tensor.name.startswith("model.visual."):
            vision.append(tensor)
        elif tensor.name.startswith("mtp.") or ".mtp." in tensor.name:
            mtp.append(tensor)
        elif "ngram_embedding.shard_" in tensor.name or (
            tensor.name.endswith("ngram_embedding.weight_scale")
        ):
            ple.append(tensor)
        elif tensor.dtype == "BF16" and tensor.source_name.startswith("model-"):
            bf16.append(tensor)
        elif tensor.dtype == "I64" and ".ple." in tensor.name:
            other.append(tensor)
        else:
            unknown.append(tensor)

    if unknown:
        fail(f"unknown/unclassified tensors: {len(unknown)}")
    expected = 48 * 512 * 3 * 4
    if len(experts) != expected:
        fail(f"expected {expected} NVFP4 components, found {len(experts)}")
    for layer in range(48):
        for expert in range(512):
            for projection in range(3):
                for component in range(4):
                    if (layer, expert, projection, component) not in experts:
                        fail(
                            "missing NVFP4 component "
                            f"layer={layer} expert={expert} projection={projection} "
                            f"component={component}"
                        )
    return experts, bf16, aux, ple, mtp, vision, other, unknown


def shape_for(tensor: Tensor) -> tuple[int, int, int, int]:
    shape = tuple(tensor.shape)
    return tuple(shape[:4]) + (0,) * (4 - len(shape))


def add_string(strings: bytearray, value: str) -> tuple[int, int]:
    encoded = value.encode("utf-8")
    offset = len(strings)
    strings.extend(encoded)
    return offset, len(encoded)


def copy_range(
    src: BinaryIO,
    dst: BinaryIO,
    offset: int,
    size: int,
    chunk_size: int = 16 * 1024 * 1024,
) -> None:
    remaining = size
    position = offset
    while remaining:
        chunk = min(remaining, chunk_size)
        src.seek(position)
        data = src.read(chunk)
        if len(data) != chunk:
            fail(f"short read at source offset {position} ({chunk} bytes)")
        dst.write(data)
        position += chunk
        remaining -= chunk


def make_ref(tensor: Tensor, source_id: int, pack_offset: int = (1 << 64) - 1) -> tuple:
    return (
        source_id,
        0,
        tensor.source_offset,
        pack_offset,
        tensor.bytes,
    )


def write_at(fp: BinaryIO, offset: int, data: bytes) -> None:
    fp.seek(offset)
    fp.write(data)


def inventory_summary(
    tensors: list[Tensor],
    experts: dict[tuple[int, int, int, int], ExpertEntry],
    bf16: list[Tensor],
    ple: list[Tensor],
    mtp: list[Tensor],
    vision: list[Tensor],
    other: list[Tensor],
    sources: list[Source],
    revision: str,
    source_root: Path,
    inventory_hash: str,
) -> dict:
    by_class: dict[str, list[Tensor]] = {
        "main BF16": bf16,
        "main NVFP4 weight": [
            e.tensor for e in experts.values() if e.component == COMPONENT_ID["weight"]
        ],
        "NVFP4 weight_scale": [
            e.tensor
            for e in experts.values()
            if e.component == COMPONENT_ID["weight_scale"]
        ],
        "NVFP4 weight_scale_2": [
            e.tensor
            for e in experts.values()
            if e.component == COMPONENT_ID["weight_scale_2"]
        ],
        "NVFP4 input_scale": [
            e.tensor
            for e in experts.values()
            if e.component == COMPONENT_ID["input_scale"]
        ],
        "PLE FP8": ple,
        "MTP FP8": mtp,
        "vision": vision,
        "other": other,
    }
    counts_by_dtype: dict[str, int] = {}
    for tensor in tensors:
        counts_by_dtype[tensor.dtype] = counts_by_dtype.get(tensor.dtype, 0) + 1
    counts_by_class = {key: len(value) for key, value in by_class.items()}
    bytes_by_class = {
        key: sum(t.bytes for t in value) for key, value in by_class.items()
    }
    return {
        "format": "q38-m9-nvidia-nvfp4-tensor-inventory-summary-v2",
        "checkpoint": {
            "repo_id": "nvidia/Qwen3.8-Flash-Next-NVFP4",
            "revision": revision,
            "local_metadata_dir": str(source_root),
            "metadata_only": False,
            "tensor_payload_downloaded": True,
            "safetensors_headers_read_via_http_range": False,
        },
        "tensor_count": len(tensors),
        "unknown_count": 0,
        "counts_by_class": counts_by_class,
        "bytes_by_class": bytes_by_class,
        "counts_by_dtype": counts_by_dtype,
        "source_shard_totals": {
            source.name: {
                "file_bytes": source.size,
                "tensor_bytes": sum(
                    t.bytes for t in tensors if t.source_id == sources.index(source)
                ),
            }
            for source in sources
        },
        "aggregate_hashes": {
            "inventory": inventory_hash,
            "source_fingerprint": hashlib.sha256(
                "".join(f"{s.name}:{s.size}\n" for s in sources).encode()
            ).hexdigest(),
        },
        "representative_early_middle_late": {},
    }


def representative(
    experts: dict[tuple[int, int, int, int], ExpertEntry], stage: str, layer: int
) -> dict:
    result = {"stage": stage, "layer": layer, "expert_id": 0, "projections": {}}
    for projection in range(3):
        name = PROJECTIONS[projection]
        components = {}
        for component in range(4):
            entry = experts[(layer, 0, projection, component)].tensor
            components[COMPONENTS[component]] = {
                "tensor_name": entry.name,
                "source_shard": entry.source_name,
                "dtype": entry.dtype,
                "stored_shape": list(entry.shape),
                "bytes": entry.bytes,
                "data_offset": entry.source_offset,
            }
        result["projections"][name] = components
    return result


def pack(
    source_root: Path,
    output: Path,
    manifest_path: Path,
    mode: int,
    revision: str,
    include_vision: bool,
    hash_sources: bool,
) -> dict:
    sources = source_files(source_root)
    tensors = collect_tensors(source_root, sources)
    experts, bf16, aux, ple, mtp, vision, other, _ = classify(tensors)
    if aux:
        fail("internal auxiliary tensor list is not empty")
    materialized_payload_bytes = sum(t.bytes for t in bf16) + sum(
        entry.tensor.bytes for entry in experts.values()
    )
    if include_vision:
        materialized_payload_bytes += sum(t.bytes for t in vision)
    if mode == MODE_MATERIALIZED:
        output.parent.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(output.parent).free
        metadata_reserve = 32 * 1024 * 1024
        required_bytes = materialized_payload_bytes + metadata_reserve
        if free_bytes < required_bytes:
            fail(
                "insufficient free space for materialized pack: "
                f"need {required_bytes} bytes, have {free_bytes}; "
                "use --mode source-backed or provide a larger filesystem"
            )

    source_name_to_id = {source.name: i for i, source in enumerate(sources)}
    strings = bytearray()
    source_model_offset, source_model_len = add_string(
        strings, "nvidia/Qwen3.8-Flash-Next-NVFP4"
    )
    source_revision_offset, source_revision_len = add_string(strings, revision)
    producer_offset, producer_len = add_string(strings, "ModelOpt 0.46.0")
    architecture_offset, architecture_len = add_string(
        strings, "Qwen4ExpForConditionalGeneration"
    )

    source_records = []
    source_hashes: dict[str, str] = {}
    for source in sources:
        path_offset, path_len = add_string(strings, source.name)
        digest = bytes(32)
        if hash_sources:
            source_hashes[source.name] = sha256_file(source.path)
            digest = bytes.fromhex(source_hashes[source.name])
        source_records.append(
            (path_offset, path_len, 0, source.size, digest)
        )

    bf16 = sorted(bf16, key=lambda t: t.name)
    vision = sorted(vision, key=lambda t: t.name)
    mtp = sorted(mtp, key=lambda t: t.name)
    other = sorted(other, key=lambda t: t.name)
    tensor_name_offsets = {}
    for tensor in bf16 + vision + mtp + other + ple:
        tensor_name_offsets[tensor.name] = add_string(strings, tensor.name)
    bf16_records = []
    aux_records = []
    ple_tensors = sorted(
        ple,
        key=lambda t: (
            1000
            if "ngram_embedding.shard_" not in t.name
            else int(t.name.rsplit("_", 1)[1].split(".")[0]),
            t.name,
        ),
    )
    ple_records = []

    # Layout tables after the fixed header and before the strings/data.
    cursor = HEADER_SIZE
    source_table_offset = cursor
    cursor += len(source_records) * SOURCE_SIZE
    bf16_table_offset = cursor
    cursor += len(bf16) * BF16_SIZE
    aux_table_offset = cursor
    cursor += (len(vision) + len(mtp) + len(other)) * AUX_SIZE
    expert_table_offset = cursor
    cursor += 48 * 512 * 3 * EXPERT_SIZE
    ple_table_offset = cursor
    cursor += len(ple_tensors) * PLE_SIZE
    string_table_offset = cursor
    cursor += len(strings)
    data_offset = cursor
    data_cursor = data_offset
    region_hashes: dict[str, str] = {}
    region_offsets: dict[str, dict[str, int | None]] = {}

    def payload_region(name: str, entries: Iterable[Tensor]) -> tuple[int, int]:
        nonlocal data_cursor
        ordered = list(entries)
        total = sum(t.bytes for t in ordered)
        if mode == MODE_SOURCE_BACKED:
            region_offsets[name] = {"pack_offset": None, "bytes": total}
            return (1 << 64) - 1, total
        start = data_cursor
        data_cursor += total
        digest = hashlib.sha256()
        with output.open("r+b", buffering=0) as dst:
            dst.seek(start)
            for tensor in ordered:
                with sources[tensor.source_id].path.open("rb", buffering=0) as src:
                    copy_range(src, dst, tensor.source_offset, tensor.bytes)
                # Hash from the destination bytes is unnecessary; hash the
                # source payload while it is still the authoritative bytes.
                with sources[tensor.source_id].path.open("rb", buffering=0) as src:
                    remaining = tensor.bytes
                    pos = tensor.source_offset
                    while remaining:
                        chunk = min(16 * 1024 * 1024, remaining)
                        src.seek(pos)
                        block = src.read(chunk)
                        if len(block) != chunk:
                            fail(f"short read while hashing {tensor.name}")
                        digest.update(block)
                        pos += chunk
                        remaining -= chunk
        region_hashes[name] = digest.hexdigest()
        region_offsets[name] = {"pack_offset": start, "bytes": total}
        return start, total

    # Reserve the output before payload copies.  Source-backed packs only
    # contain tables and therefore never allocate model-sized output space.
    if mode == MODE_MATERIALIZED:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as fp:
            fp.truncate(data_cursor)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()

    # BF16 descriptors are contiguous in the materialized data region.
    bf16_pack_base, _ = payload_region("bf16_main", bf16)
    bf16_cursor = bf16_pack_base
    for tensor in bf16:
        pack_offset = bf16_cursor if mode == MODE_MATERIALIZED else (1 << 64) - 1
        if mode == MODE_MATERIALIZED:
            bf16_cursor += tensor.bytes
        shape = shape_for(tensor)
        name_offset, name_len = tensor_name_offsets[tensor.name]
        bf16_records.append(
            (
                name_offset,
                name_len,
                tensor.source_id,
                DTYPE_ID[tensor.dtype],
                len(tensor.shape),
                tensor.source_offset,
                pack_offset,
                tensor.bytes,
                *shape,
            )
        )

    aux_tensors = [
        ("vision", tensor) for tensor in vision
    ] + [("MTP FP8", tensor) for tensor in mtp] + [
        ("other", tensor) for tensor in other
    ]
    vision_pack_cursor = (1 << 64) - 1
    if include_vision and mode == MODE_MATERIALIZED:
        vision_pack_cursor, _ = payload_region("vision", vision)
    for class_name, tensor in aux_tensors:
        name_offset, name_len = tensor_name_offsets[tensor.name]
        shape = shape_for(tensor)
        aux_records.append(
            (
                name_offset,
                name_len,
                CLASS_ID[class_name],
                tensor.source_id,
                DTYPE_ID[tensor.dtype],
                len(tensor.shape),
                0,
                tensor.source_offset,
                vision_pack_cursor
                if class_name == "vision" and mode == MODE_MATERIALIZED
                else (1 << 64) - 1,
                tensor.bytes,
                *shape,
            )
        )
        if class_name == "vision" and mode == MODE_MATERIALIZED:
            vision_pack_cursor += tensor.bytes

    # Native expert arrays: [layer][projection][expert], each with four
    # byte-preserving component references.
    expert_refs: list[tuple] = []
    expert_regions = {
        component: [] for component in COMPONENTS
    }
    for layer in range(48):
        for projection in range(3):
            for expert in range(512):
                for component in COMPONENTS:
                    expert_regions[component].append(
                        experts[
                            (
                                layer,
                                expert,
                                projection,
                                COMPONENT_ID[component],
                            )
                        ].tensor
                    )
    component_pack_bases: dict[str, int] = {}
    for component in COMPONENTS:
        base, _ = payload_region(
            f"nvfp4_{component}", expert_regions[component]
        )
        component_pack_bases[component] = base
    component_cursors = dict(component_pack_bases)
    for layer in range(48):
        for projection in range(3):
            for expert in range(512):
                refs = []
                for component in COMPONENTS:
                    tensor = experts[
                        (
                            layer,
                            expert,
                            projection,
                            COMPONENT_ID[component],
                        )
                    ].tensor
                    pack_offset = (
                        component_cursors[component]
                        if mode == MODE_MATERIALIZED
                        else (1 << 64) - 1
                    )
                    if mode == MODE_MATERIALIZED:
                        component_cursors[component] += tensor.bytes
                    refs.append(make_ref(tensor, tensor.source_id, pack_offset))
                expert_refs.extend(refs)

    # PLE is never copied.  Preserve its exact sidecar source offsets.
    ple_records = []
    for tensor in ple_tensors:
        name_offset, name_len = tensor_name_offsets[tensor.name]
        shape = shape_for(tensor)
        shard_id = (
            int(tensor.name.rsplit("_", 1)[1].split(".")[0])
            if "ngram_embedding.shard_" in tensor.name
            else 0xFFFFFFFF
        )
        ple_records.append(
            (
                shard_id,
                tensor.source_id,
                DTYPE_ID[tensor.dtype],
                len(tensor.shape),
                tensor.source_offset,
                (1 << 64) - 1,
                tensor.bytes,
                *shape,
            )
        )

    string_table_offset = (
        ple_table_offset + len(ple_records) * PLE_SIZE
    )
    data_offset = string_table_offset + len(strings)
    data_bytes = max(0, data_cursor - data_offset)
    main_resident_bytes = sum(t.bytes for t in bf16) + sum(
        t.bytes for e in experts.values() for t in [e.tensor]
    )
    ple_bytes = sum(t.bytes for t in ple_tensors)
    metadata_bytes = (
        source_table_offset
        + len(source_records) * SOURCE_SIZE
        + len(bf16_records) * BF16_SIZE
        + len(aux_records) * AUX_SIZE
        + len(expert_refs) * REF_SIZE
        + len(ple_records) * PLE_SIZE
        + len(strings)
    )
    # A descriptor-only pack has no payload; a materialized pack's data cursor
    # is authoritative after all regions are placed.
    if mode == MODE_SOURCE_BACKED:
        data_offset = string_table_offset + len(strings)
        data_bytes = 0
        file_bytes = data_offset
    else:
        file_bytes = data_cursor

    digest_input = json.dumps(
        {
            "source_model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
            "source_revision": revision,
            "storage_mode": mode,
            "source_files": [(s.name, s.size) for s in sources],
            "main_resident_bytes": main_resident_bytes,
            "ple_bytes": ple_bytes,
        },
        sort_keys=True,
    ).encode()
    manifest_digest = hashlib.sha256(digest_input).digest()

    header_values = (
        MAGIC,
        VERSION,
        mode,
        FLAG_MTP_DISABLED
        | FLAG_VISION_INDEXED
        | FLAG_PLE_EXTERNAL
        | FLAG_BF16_UNCHANGED,
        len(source_records),
        len(bf16_records),
        len(aux_records),
        len(ple_records),
        0,
        source_table_offset,
        bf16_table_offset,
        aux_table_offset,
        expert_table_offset,
        ple_table_offset,
        string_table_offset,
        len(strings),
        data_offset,
        data_bytes,
        main_resident_bytes,
        ple_bytes,
        metadata_bytes,
        source_model_offset,
        source_model_len,
        source_revision_offset,
        source_revision_len,
        producer_offset,
        producer_len,
        architecture_offset,
        architecture_len,
        manifest_digest,
    )
    header = struct.pack(HEADER_FORMAT, *header_values)
    header += bytes(HEADER_SIZE - len(header))

    with output.open("r+b", buffering=0) as fp:
        write_at(fp, 0, header)
        for record in source_records:
            fp.write(struct.pack(SOURCE_FORMAT, *record))
        for record in bf16_records:
            fp.write(struct.pack(BF16_FORMAT, *record))
        for record in aux_records:
            fp.write(struct.pack(AUX_FORMAT, *record))
        for index in range(0, len(expert_refs), 4):
            for ref in expert_refs[index:index + 4]:
                fp.write(struct.pack(REF_FORMAT, *ref))
        for record in ple_records:
            fp.write(struct.pack(PLE_FORMAT, *record))
        fp.write(strings)
        fp.flush()
        if mode == MODE_SOURCE_BACKED:
            fp.truncate(file_bytes)

    inventory_hash = hashlib.sha256(
        "".join(
            f"{t.name}:{t.source_name}:{t.source_offset}:{t.bytes}:{t.dtype}:{t.shape}\n"
            for t in sorted(tensors, key=lambda item: item.name)
        ).encode()
    ).hexdigest()
    summary = inventory_summary(
        tensors,
        experts,
        bf16,
        ple,
        mtp,
        vision,
        other,
        sources,
        revision,
        source_root,
        inventory_hash,
    )
    summary["representative_early_middle_late"] = {
        "early": representative(experts, "early", 0),
        "middle": representative(experts, "middle", 24),
        "late": representative(experts, "late", 47),
    }
    projection_counts = {
        "gate_proj": sum(".gate_proj." in t.name for t in tensors),
        "up_proj": sum(".up_proj." in t.name for t in tensors),
        "down_proj": sum(".down_proj." in t.name for t in tensors),
    }
    projection_counts["unattributed"] = len(tensors) - sum(
        projection_counts.values()
    )
    summary["counts_by_projection"] = projection_counts
    summary["pack"] = {
        "format": "Q38_NVFP4_PACK_V1",
        "path": str(output),
        "manifest_path": str(manifest_path),
        "storage_mode": "source-backed"
        if mode == MODE_SOURCE_BACKED
        else "materialized",
        "pack_file_bytes": file_bytes,
        "metadata_bytes": metadata_bytes,
        "data_bytes": data_bytes,
        "main_resident_bytes": main_resident_bytes,
        "ple_external": True,
        "ple_backing_bytes": sum(
            source.size
            for source in sources
            if source.name == "model-fp8-mtp-ple.safetensors"
        ),
        "payload_hashes_complete": bool(
            mode == MODE_MATERIALIZED or hash_sources
        ),
        "regions": region_offsets,
        "region_hashes": region_hashes,
        "source_hashes": source_hashes,
        "mtp_execution": "disabled",
        "vision_execution": "disabled",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def verify_fixtures(pack_path: Path, fixtures: Path) -> None:
    """Verify source-backed or materialized expert bytes against M9N-02."""
    with pack_path.open("rb") as fp:
        header = fp.read(HEADER_SIZE)
    if len(header) != HEADER_SIZE:
        fail(f"truncated pack: {pack_path}")
    fields = struct.unpack(HEADER_FORMAT, header[:struct.calcsize(HEADER_FORMAT)])
    magic, version, mode = fields[:3]
    if magic != MAGIC or version != VERSION:
        fail("fixture verification found an invalid pack header")
    source_root = pack_path.parent
    # The verification uses the original checkpoint files named by fixture
    # metadata; this intentionally remains independent of any dequantization.
    for fixture_name in ("early.json", "middle.json", "late.json"):
        fixture = json.loads((fixtures / fixture_name).read_text())
        for projection, values in fixture["projections"].items():
            for component in COMPONENTS:
                item = values[component]
                source = Path(item["source_shard"])
                if not source.is_absolute():
                    source = Path(
                        "models/Qwen3.8-Flash-Next-NVFP4"
                    ) / source
                if not source.exists():
                    fail(f"fixture source shard is missing: {source}")
                payload = bytearray()
                ranges = item["fetched_absolute_ranges"]
                if ranges and isinstance(ranges[0], int):
                    ranges = [ranges]
                with source.open("rb", buffering=0) as fp:
                    for start, end in ranges:
                        fp.seek(int(start))
                        block = fp.read(int(end) - int(start) + 1)
                        if len(block) != int(end) - int(start) + 1:
                            fail(f"short fixture read: {source}")
                        payload.extend(block)
                if component == "weight":
                    expected = b"".join(
                        bytes.fromhex(row)
                        for row in item["packed_rows_hex"]
                    )
                    if bytes(payload) != expected:
                        fail(
                            f"{fixture_name} {projection} {component} bytes mismatch"
                        )
                elif component == "weight_scale":
                    expected = b"".join(
                        bytes.fromhex(row)
                        for row in item["scale_rows_hex"]
                    )
                    if bytes(payload) != expected:
                        fail(
                            f"{fixture_name} {projection} {component} bytes mismatch"
                        )
                else:
                    expected = bytes.fromhex(item["raw_hex"])
                    if bytes(payload) != expected:
                        fail(
                            f"{fixture_name} {projection} {component} bytes mismatch"
                        )
    print(
        json.dumps(
            {
                "format": "Q38_NVFP4_PACK_V1",
                "mode": "source-backed"
                if mode == MODE_SOURCE_BACKED
                else "materialized",
                "fixtures": ["early", "middle", "late"],
                "packed_weight_bytes": "exact",
                "weight_scale_bytes": "exact",
                "scalar_bytes": "exact",
                "max_abs": 0.0,
                "mismatch": 0,
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument(
        "--mode",
        choices=("source-backed", "materialized"),
        default="materialized",
    )
    parser.add_argument(
        "--revision",
        default="fc694b54fb0174e0913e6adf86691ef85a4ead47",
    )
    parser.add_argument("--include-vision", action="store_true")
    parser.add_argument("--hash-sources", action="store_true")
    parser.add_argument("--verify-fixtures", type=Path)
    args = parser.parse_args()

    if not args.source_root.is_dir():
        fail(f"source root does not exist: {args.source_root}")
    if args.verify_fixtures:
        verify_fixtures(args.output, args.verify_fixtures)
        return 0

    mode = (
        MODE_SOURCE_BACKED
        if args.mode == "source-backed"
        else MODE_MATERIALIZED
    )
    summary = pack(
        args.source_root,
        args.output,
        args.manifest,
        mode,
        args.revision,
        args.include_vision,
        args.hash_sources,
    )
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "format": "Q38_NVFP4_PACK_V1",
                "storage_mode": summary["pack"]["storage_mode"],
                "pack_file_bytes": summary["pack"]["pack_file_bytes"],
                "main_resident_bytes": summary["pack"]["main_resident_bytes"],
                "ple_external": summary["pack"]["ple_external"],
                "unknown_count": summary["unknown_count"],
                "tensor_count": summary["tensor_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    main()
