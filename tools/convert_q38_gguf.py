#!/usr/bin/env python3
"""Write a deterministic, runtime-only GGUF from the frozen source inventory.

With --quantize, routed gate/up and down tensors are passed through the donor
block writers.  The source shards remain mmap/read streamed; no model-sized
dequantized Python buffer is created.
"""

import argparse
import json
import mmap
import os
import re
import struct
import subprocess


GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_VALUE_UINT32 = 4
GGUF_VALUE_STRING = 8
GGUF_VALUE_BOOL = 7
GGUF_TYPE_I64 = 27
GGUF_TYPE_BF16 = 30
GGUF_TYPES = {"Q2_K": 10, "IQ2_XXS": 16, "Q4_K": 12, "Q8_0": 8,
              "BF16": GGUF_TYPE_BF16, "I64": GGUF_TYPE_I64}
BLOCK_INFO = {"Q2_K": (256, 84), "IQ2_XXS": (256, 66),
              "Q4_K": (256, 144), "Q8_0": (32, 34)}
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
        stream.write(struct.pack("<B", int(value)))
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
    if name.startswith("model.language_model.hyper_connection_mixer."):
        return True
    marker = ".layers."
    if marker not in name:
        return False
    prefix, suffix = name.split(marker, 1)
    layer_text = suffix.split(".", 1)[0]
    return (prefix == "model.language_model" and layer_text.isdigit()
            and int(layer_text) <= max_layer)


def load_rules(manifest):
    rules = []
    for rule in manifest["rules"]:
        pattern = re.compile(rule["pattern"])
        rules.append((rule, pattern))
    return rules


def rule_for(tensor, rules):
    matches = [(rule, pattern) for rule, pattern in rules
               if rule["class"] == tensor["class"]
               and pattern.search(tensor["name"])]
    if len(matches) != 1:
        raise ValueError(f"{tensor['name']}: expected one manifest rule, got "
                         f"{len(matches)}")
    return matches[0][0]


def transformed_shape(tensor, rule, quantize):
    shape = list(tensor["shape"])
    if quantize and rule.get("layout_transform") == "transpose_last_two_axes":
        if len(shape) < 2:
            raise ValueError(f"{tensor['name']}: cannot transpose shape")
        shape[-1], shape[-2] = shape[-2], shape[-1]
    return shape


def output_type(tensor, rule, quantize, shape):
    if not quantize or tensor["source_dtype"] != "BF16":
        return tensor["source_dtype"]
    target = rule["quant_type"]
    if target in BLOCK_INFO and (len(shape) < 2 or shape[-1] % BLOCK_INFO[target][0]):
        target = rule.get("fallback_quant_type", tensor["source_dtype"])
    return target


def source_range(model_dir, tensor):
    shard_path = os.path.join(model_dir, tensor["source_shard"])
    with open(shard_path, "rb") as shard:
        header_length_raw = shard.read(8)
        if len(header_length_raw) != 8:
            raise ValueError(f"{shard_path}: missing header length")
        header_length = struct.unpack("<Q", header_length_raw)[0]
    start = 8 + header_length + tensor["data_offsets"][0]
    return shard_path, start


def copy_source(stream, model_dir, tensor):
    shard_path, start = source_range(model_dir, tensor)
    remaining = tensor["bytes_source"]
    with open(shard_path, "rb") as shard:
        shard.seek(start)
        with mmap.mmap(shard.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            position = start
            while remaining:
                chunk = min(8 * 1024 * 1024, remaining)
                stream.write(mapped[position:position + chunk])
                position += chunk
                remaining -= chunk


def quantize_source(stream, model_dir, tensor, rule, shape, quantizer, part_path):
    shard_path, start = source_range(model_dir, tensor)
    command = [quantizer, "--input", shard_path, "--output", part_path,
               "--offset", str(start), "--bytes", str(tensor["bytes_source"]),
               "--type", rule["quant_type"], "--shape"] + [str(x) for x in tensor["shape"]]
    if rule.get("layout_transform") == "transpose_last_two_axes":
        command.append("--transpose-last-two")
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        with open(part_path, "rb") as part:
            while True:
                chunk = part.read(8 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
    finally:
        try:
            os.unlink(part_path)
        except FileNotFoundError:
            pass


def estimate_bytes(tensor, target, shape):
    elements = 1
    for dimension in shape:
        elements *= dimension
    if target in BLOCK_INFO:
        block, block_bytes = BLOCK_INFO[target]
        if shape[-1] % block:
            raise ValueError(f"{tensor['name']}: final dimension is not block aligned")
        rows = elements // shape[-1]
        return rows * (shape[-1] // block) * block_bytes
    return elements * (2 if target == "BF16" else 8)


def write_subset(model_dir, inventory_path, output_path, max_layer, revision,
                 classes_path=None, manifest_path=None, quantize=False,
                 quantizer="tools/q38_quantize", plan_output=None):
    with open(inventory_path, encoding="utf-8") as stream:
        source = json.load(stream)
    if classes_path:
        with open(classes_path, encoding="utf-8") as stream:
            classes = json.load(stream)
        by_name = {tensor["name"]: tensor for tensor in classes["tensors"]}
    else:
        by_name = {tensor["name"]: tensor for tensor in source["tensors"]}
    if manifest_path:
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        rules = load_rules(manifest)
    else:
        manifest = None
        rules = []

    tensors = [by_name[t["name"]] for t in source["tensors"]
               if selected(t, max_layer) and t.get("included_runtime", True)]
    tensors.sort(key=lambda tensor: tensor["name"])
    if not tensors:
        raise ValueError("subset selection produced no tensors")

    entries = []
    for tensor in tensors:
        rule = rule_for(tensor, rules) if rules else {
            "quant_type": tensor["source_dtype"], "include": True}
        if not rule.get("include", True):
            raise ValueError(f"{tensor['name']}: selected tensor is excluded")
        shape = transformed_shape(tensor, rule, quantize)
        target = output_type(tensor, rule, quantize, shape)
        if target not in GGUF_TYPES:
            raise ValueError(f"{tensor['name']}: unsupported output type {target}")
        if (target in ("Q2_K", "IQ2_XXS", "Q4_K", "Q8_0")
                and tensor["source_dtype"] != "BF16"):
            raise ValueError(f"{tensor['name']}: quantizer requires BF16 source")
        entries.append((tensor, rule, target, shape))

    estimate = sum(estimate_bytes(tensor, target, shape)
                   for tensor, _, target, shape in entries)
    if plan_output:
        plan = {
            "format": "q38_runtime_conversion_plan_v1",
            "max_layer": max_layer,
            "quantized": quantize,
            "tensor_count": len(entries),
            "estimated_tensor_bytes": estimate,
            "source_bytes": sum(t["bytes_source"] for t, _, _, _ in entries),
            "headroom_target_bytes": 108 * 1024 ** 3,
            "status": "pass" if estimate <= 108 * 1024 ** 3 else "blocked",
            "reason": ("estimated payload exceeds the M1 108 GiB pressure target"
                       if estimate > 108 * 1024 ** 3 else ""),
        }
        with open(plan_output, "w", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return

    metadata = [
        ("general.name", GGUF_VALUE_STRING,
         "Qwen3.8-Flash-Next runtime-only "
         + ("Q2Experts-BF16Core-BF16PLE" if quantize else "BF16 subset")),
        ("general.architecture", GGUF_VALUE_STRING, "qwen4_exp"),
        ("general.alignment", GGUF_VALUE_UINT32, ALIGNMENT),
        ("q38.runtime_only", GGUF_VALUE_BOOL, True),
        ("q38.max_layer", GGUF_VALUE_UINT32, max_layer),
        ("q38.excluded_vision", GGUF_VALUE_BOOL, True),
        ("q38.excluded_mtp", GGUF_VALUE_BOOL, True),
        ("q38.quantized", GGUF_VALUE_BOOL, quantize),
        ("q38.down_proj_layout", GGUF_VALUE_STRING,
         "transpose_last_two_axes" if quantize else "source_layout"),
        ("q38.source_revision", GGUF_VALUE_STRING, revision),
    ]
    if manifest:
        metadata.append(("q38.quant_manifest", GGUF_VALUE_STRING,
                         os.path.basename(manifest_path)))

    with open(output_path, "wb") as output:
        output.write(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION,
                                 len(entries), len(metadata)))
        for key, kind, value in metadata:
            put_kv(output, key, kind, value)
        offsets = []
        for tensor, rule, target, shape in entries:
            put_string(output, tensor["name"])
            output.write(struct.pack("<I", len(shape)))
            for dimension in shape:
                output.write(struct.pack("<Q", dimension))
            output.write(struct.pack("<I", GGUF_TYPES[target]))
            offsets.append((tensor, rule, target, output.tell()))
            output.write(struct.pack("<Q", 0))
        align(output, ALIGNMENT)
        data_start = output.tell()
        for index, (tensor, rule, target, info_pos) in enumerate(offsets):
            relative_offset = output.tell() - data_start
            output_pos = info_pos
            output.seek(output_pos)
            output.write(struct.pack("<Q", relative_offset))
            output.seek(0, os.SEEK_END)
            align(output, ALIGNMENT)
            if quantize and target in ("Q2_K", "IQ2_XXS", "Q4_K", "Q8_0"):
                part_path = output_path + f".tensor-{index}.part"
                quantize_source(output, model_dir, tensor, rule,
                                transformed_shape(tensor, rule, True),
                                quantizer, part_path)
            else:
                copy_source(output, model_dir, tensor)
            align(output, ALIGNMENT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/home/lvx/q38model")
    parser.add_argument("--inventory", default="artifacts/m1/source_inventory.json")
    parser.add_argument("--classes")
    parser.add_argument("--manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-layer", type=int, default=3)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--quantizer", default="tools/q38_quantize")
    parser.add_argument("--plan-output")
    args = parser.parse_args()
    write_subset(args.model_dir, args.inventory, args.output, args.max_layer,
                 args.revision, args.classes, args.manifest, args.quantize,
                 args.quantizer, args.plan_output)


if __name__ == "__main__":
    main()
