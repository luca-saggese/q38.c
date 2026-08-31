#!/usr/bin/env python3
"""Convert a strict text-only Qwen subset to a mmap-friendly GGUF file."""

import argparse
import json
import mmap
import os
import struct


GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_VALUE_UINT32 = 4
GGUF_VALUE_STRING = 8
GGUF_VALUE_BOOL = 7
GGUF_TYPE_BF16 = 30
GGUF_TYPE_I64 = 27
ALIGNMENT = 32


def put_string(stream, value):
    encoded = value.encode("utf-8")
    stream.write(struct.pack("<Q", len(encoded)))
    stream.write(encoded)


def put_kv(stream, key, kind, value):
    put_string(stream, key)
    stream.write(struct.pack("<I", kind))
    if kind == GGUF_VALUE_STRING:
        put_string(stream, value)
    elif kind == GGUF_VALUE_UINT32:
        stream.write(struct.pack("<I", value))
    elif kind == GGUF_VALUE_BOOL:
        stream.write(struct.pack("<B", value))
    else:
        raise ValueError(f"unsupported metadata type: {kind}")


def align(stream, alignment):
    pad = (-stream.tell()) % alignment
    if pad:
        stream.write(b"\0" * pad)


def selected(tensor, max_layer):
    name = tensor["name"]
    if name in ("lm_head.weight", "model.language_model.embed_tokens.weight"):
        return True
    marker = ".layers."
    if marker not in name:
        return False
    prefix, suffix = name.split(marker, 1)
    layer_text = suffix.split(".", 1)[0]
    return prefix == "model.language_model" and layer_text.isdigit() and int(layer_text) <= max_layer


def write_subset(model_dir, inventory_path, output_path, max_layer, revision):
    with open(inventory_path, encoding="utf-8") as stream:
        inventory = json.load(stream)
    tensors = [t for t in inventory["tensors"] if selected(t, max_layer)]
    tensors.sort(key=lambda tensor: tensor["name"])
    if not tensors:
        raise ValueError("subset selection produced no tensors")
    source_types = {"BF16": GGUF_TYPE_BF16, "I64": GGUF_TYPE_I64}
    unsupported = sorted({t["source_dtype"] for t in tensors} - source_types.keys())
    if unsupported:
        raise ValueError(
            "subset contains unsupported source dtypes: "
            + ", ".join(unsupported)
        )

    metadata = [
        ("general.name", GGUF_VALUE_STRING, "Qwen3.8-Flash-Next runtime-only BF16 subset"),
        ("general.architecture", GGUF_VALUE_STRING, "qwen4_exp"),
        ("general.alignment", GGUF_VALUE_UINT32, ALIGNMENT),
        ("q38.runtime_only", GGUF_VALUE_BOOL, True),
        ("q38.max_layer", GGUF_VALUE_UINT32, max_layer),
        ("q38.excluded_vision", GGUF_VALUE_BOOL, True),
        ("q38.excluded_mtp", GGUF_VALUE_BOOL, True),
        ("q38.source_revision", GGUF_VALUE_STRING, revision),
    ]

    header_size = 24
    for key, kind, value in metadata:
        encoded = key.encode("utf-8")
        header_size += 8 + len(encoded) + 4
        if kind == GGUF_VALUE_STRING:
            encoded_value = value.encode("utf-8")
            header_size += 8 + len(encoded_value)
        elif kind == GGUF_VALUE_UINT32:
            header_size += 4
        elif kind == GGUF_VALUE_BOOL:
            header_size += 1
    for tensor in tensors:
        header_size += 8 + len(tensor["name"].encode("utf-8"))
        header_size += 4 + 8 * len(tensor["shape"]) + 4 + 8

    with open(output_path, "wb") as output:
        output.write(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION,
                                 len(tensors), len(metadata)))
        for key, kind, value in metadata:
            put_kv(output, key, kind, value)
        offsets = []
        for tensor in tensors:
            put_string(output, tensor["name"])
            output.write(struct.pack("<I", len(tensor["shape"])))
            for dimension in tensor["shape"]:
                output.write(struct.pack("<Q", dimension))
            output.write(struct.pack("<I", source_types[tensor["source_dtype"]]))
            offsets.append((tensor, output.tell()))
            output.write(struct.pack("<Q", 0))
        align(output, ALIGNMENT)
        data_start = output.tell()
        for tensor, info_pos in offsets:
            relative_offset = output.tell() - data_start
            output_pos = info_pos
            output.seek(output_pos)
            output.write(struct.pack("<Q", relative_offset))
            output.seek(0, os.SEEK_END)
            align(output, ALIGNMENT)
            shard_path = os.path.join(model_dir, tensor["source_shard"])
            with open(shard_path, "rb") as shard:
                header_length = struct.unpack("<Q", shard.read(8))[0]
                source_offset = 8 + header_length + tensor["data_offsets"][0]
                shard.seek(source_offset)
                remaining = tensor["bytes_source"]
                with mmap.mmap(shard.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                    position = source_offset
                    while remaining:
                        chunk = min(8 * 1024 * 1024, remaining)
                        output.write(mapped[position:position + chunk])
                        position += chunk
                        remaining -= chunk
            align(output, ALIGNMENT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/home/lvx/q38model")
    parser.add_argument("--inventory", default="artifacts/m1/source_inventory.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-layer", type=int, default=3)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    write_subset(args.model_dir, args.inventory, args.output, args.max_layer, args.revision)


if __name__ == "__main__":
    main()
