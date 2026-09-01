#!/usr/bin/env python3
"""Preflight a file-backed q38 runtime artifact before any forward call.

The report is deliberately tensor-granular: it compares the original
Safetensors checkpoint with the GGUF artifact and records the exact shape,
qtype, and runtime-domain support for every required tensor.  A missing or
unbindable tensor is a hard failure; this tool never invents a replacement.
"""

import argparse
import json
import os
import re
import struct
from pathlib import Path


GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84),
    12: ("Q4_K", 256, 144),
    16: ("IQ2_XXS", 256, 66),
    27: ("I64", 1, 8),
    30: ("BF16", 1, 2),
}
SAFETENSOR_BYTES = {"BF16": 2, "I64": 8, "F16": 2, "F32": 4}


class PreflightError(Exception):
    """A checkpoint or artifact cannot be proven usable."""


def read_safetensors_header(path):
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise PreflightError(f"{path}: missing Safetensors header length")
        header_size = struct.unpack("<Q", raw)[0]
        if header_size > 128 * 1024 * 1024:
            raise PreflightError(f"{path}: unreasonable Safetensors header")
        header_raw = stream.read(header_size)
        if len(header_raw) != header_size:
            raise PreflightError(f"{path}: truncated Safetensors header")
    try:
        return json.loads(header_raw)
    except json.JSONDecodeError as exc:
        raise PreflightError(f"{path}: invalid Safetensors JSON: {exc}") from exc


def load_checkpoint(model_dir):
    index_path = model_dir / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"{index_path}: cannot read index: {exc}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PreflightError(f"{index_path}: empty or invalid weight_map")

    headers = {}
    for shard_name in sorted(set(weight_map.values())):
        shard = model_dir / shard_name
        if not shard.is_file():
            raise PreflightError(f"missing checkpoint shard: {shard}")
        headers[shard_name] = read_safetensors_header(shard)

    tensors = {}
    shard_errors = []
    for name, shard_name in sorted(weight_map.items()):
        entry = headers[shard_name].get(name)
        if not isinstance(entry, dict):
            shard_errors.append(f"{name}: missing entry in {shard_name}")
            continue
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        dtype = entry.get("dtype")
        if (not isinstance(shape, list) or
                any(not isinstance(dim, int) or dim < 0 for dim in shape) or
                not isinstance(offsets, list) or len(offsets) != 2 or
                not all(isinstance(value, int) for value in offsets) or
                offsets[1] < offsets[0] or dtype not in SAFETENSOR_BYTES):
            shard_errors.append(f"{name}: invalid shape, dtype, or data_offsets")
            continue
        elements = 1
        for dim in shape:
            elements *= dim
        source_bytes = offsets[1] - offsets[0]
        expected_bytes = elements * SAFETENSOR_BYTES[dtype]
        shard_size = (model_dir / shard_name).stat().st_size
        # The exact serialized header size is read separately below; offsets
        # are relative to the tensor-data section, not to this JSON estimate.
        # Bounds are checked against the header length recorded on disk.
        with (model_dir / shard_name).open("rb") as stream:
            header_len = struct.unpack("<Q", stream.read(8))[0]
        data_start = 8 + header_len
        if (offsets[0] < 0 or offsets[1] > shard_size - data_start or
                source_bytes != expected_bytes):
            shard_errors.append(
                f"{name}: source range/byte count invalid "
                f"(dtype={dtype}, shape={shape}, bytes={source_bytes}, "
                f"expected={expected_bytes})")
            continue
        tensors[name] = {
            "name": name,
            "shape": shape,
            "source_dtype": dtype,
            "bytes_source": source_bytes,
            "source_shard": shard_name,
        }
    if shard_errors:
        raise PreflightError("; ".join(shard_errors[:8]))
    return tensors, len(set(weight_map.values()))


class Cursor:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, count):
        if count < 0 or self.pos + count > len(self.data):
            raise PreflightError("truncated GGUF header")
        result = self.data[self.pos:self.pos + count]
        self.pos += count
        return result

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8", errors="strict")


def skip_gguf_value(cursor, kind, depth=0):
    if depth > 8:
        raise PreflightError("GGUF metadata nesting is too deep")
    scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                    7: 1, 10: 8, 11: 8, 12: 8}
    if kind in scalar_sizes:
        cursor.take(scalar_sizes[kind])
    elif kind == 8:
        cursor.take(cursor.u64())
    elif kind == 9:
        item_kind = cursor.u32()
        count = cursor.u64()
        for _ in range(count):
            skip_gguf_value(cursor, item_kind, depth + 1)
    else:
        raise PreflightError(f"unsupported GGUF metadata type {kind}")


def load_gguf(path):
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        # Header and metadata are small; cap the read so a corrupt count
        # cannot cause a model-sized allocation.
        data = stream.read(min(path.stat().st_size, 256 * 1024 * 1024))
    cursor = Cursor(data)
    if cursor.u32() != GGUF_MAGIC:
        raise PreflightError(f"{path}: invalid GGUF magic")
    if cursor.u32() != GGUF_VERSION:
        raise PreflightError(f"{path}: unsupported GGUF version")
    tensor_count = cursor.u64()
    metadata_count = cursor.u64()
    metadata = {}
    for _ in range(metadata_count):
        key = cursor.string()
        kind = cursor.u32()
        value_pos = cursor.pos
        skip_gguf_value(cursor, kind)
        if kind == 7:
            metadata[key] = bool(data[value_pos])
        elif kind == 4:
            metadata[key] = struct.unpack_from("<I", data, value_pos)[0]
        elif kind == 8:
            value_cursor = Cursor(data)
            value_cursor.pos = value_pos
            metadata[key] = value_cursor.string()
    tensors = {}
    for _ in range(tensor_count):
        name = cursor.string()
        ndim = cursor.u32()
        if ndim > 8:
            raise PreflightError(f"{name}: invalid GGUF rank {ndim}")
        shape = list(struct.unpack("<" + "Q" * ndim, cursor.take(8 * ndim)))
        kind = cursor.u32()
        rel_offset = cursor.u64()
        if name in tensors:
            raise PreflightError(f"duplicate GGUF tensor: {name}")
        tensors[name] = {
            "name": name,
            "shape": shape,
            "type": kind,
            "qtype": GGUF_TYPES.get(kind, (f"UNKNOWN({kind})", 0, 0))[0],
            "rel_offset": rel_offset,
        }
    alignment = metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment <= 0:
        raise PreflightError("GGUF has invalid general.alignment")
    data_start = (cursor.pos + alignment - 1) // alignment * alignment
    ranges = []
    for tensor in tensors.values():
        info = GGUF_TYPES.get(tensor["type"])
        if info is None:
            raise PreflightError(
                f"{tensor['name']}: unsupported GGUF tensor type "
                f"{tensor['type']}")
        block, block_bytes = info[1:]
        elements = product(tensor["shape"])
        if elements % block:
            raise PreflightError(
                f"{tensor['name']}: shape is not aligned for {tensor['qtype']}")
        tensor_bytes = elements // block * block_bytes
        absolute = data_start + tensor["rel_offset"]
        if absolute > file_size or tensor_bytes > file_size - absolute:
            raise PreflightError(
                f"{tensor['name']}: GGUF payload extends past end of artifact")
        tensor["bytes"] = tensor_bytes
        ranges.append((absolute, absolute + tensor_bytes, tensor["name"]))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise PreflightError(
                f"overlapping GGUF tensor payloads: {previous[2]} and "
                f"{current[2]}")
    return metadata, tensors


def product(shape):
    value = 1
    for dim in shape:
        value *= dim
    return value


def expected_for(tensor, manifest_rules):
    name = tensor["name"]
    matches = [
        rule for rule in manifest_rules
        if re.search(rule["pattern"], name) and rule.get("include", True)
    ]
    if len(matches) != 1:
        raise PreflightError(
            f"{name}: expected exactly one included manifest rule, got "
            f"{len(matches)}")
    rule = matches[0]
    shape = list(tensor["shape"])
    target = rule["quant_type"]
    if tensor["source_dtype"] != "BF16":
        return tensor["source_dtype"], shape, rule["id"]
    if rule.get("layout_transform") == "transpose_last_two_axes":
        shape[-1], shape[-2] = shape[-2], shape[-1]
    if target in ("Q2_K", "Q4_K", "Q8_0", "IQ2_XXS"):
        block = GGUF_TYPES[{"Q2_K": 10, "Q4_K": 12,
                            "Q8_0": 8, "IQ2_XXS": 16}[target]][1]
        if len(shape) < 2 or not shape or shape[-1] % block:
            target = rule.get("fallback_quant_type", tensor["source_dtype"])
    return target, shape, rule["id"]


def support_reason(item, expected_type, expected_shape, actual, tensor):
    if actual is None:
        return False, "tensor missing from GGUF artifact"
    if actual["shape"] != expected_shape:
        return False, "GGUF shape differs from required runtime layout"
    if actual["qtype"] != expected_type:
        return False, f"GGUF qtype {actual['qtype']} != required {expected_type}"
    allowed = {
        "BF16": {"embedding", "output", "router", "shared_expert", "gdn",
                 "qsa", "gr", "ple"},
        "I64": {"ple"},
        "Q2_K": {"routed_expert"},
        "Q8_0": {"ple"},
    }
    if expected_type not in allowed or tensor["class"] not in allowed[expected_type]:
        return False, "qtype has no supported q38 execution path for this domain"
    elements = product(expected_shape)
    block, block_bytes = GGUF_TYPES[actual["type"]][1:]
    if elements % block != 0:
        return False, "tensor element count is not divisible by qtype block size"
    expected_bytes = (elements // block) * block_bytes
    if actual.get("bytes") is not None and actual["bytes"] != expected_bytes:
        return False, "GGUF payload byte count does not match qtype and shape"
    return True, "supported by file-backed q38 binder/forward path"


def run(args):
    source, shard_count = load_checkpoint(args.model_dir)
    classes = json.loads(args.classes.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    required = [
        tensor for tensor in classes["tensors"]
        if tensor.get("included_runtime", False)
    ]
    if len(required) != 1294:
        raise PreflightError(
            f"runtime inventory contains {len(required)} tensors; expected 1294")
    for tensor in required:
        if tensor["name"] not in source:
            raise PreflightError(f"runtime tensor absent from checkpoint: {tensor['name']}")
    metadata, artifact = load_gguf(args.artifact)
    reports = []
    failures = []
    rules = manifest["rules"]
    required_names = {tensor["name"] for tensor in required}
    artifact_names = set(artifact)
    for tensor in required:
        expected_type, expected_shape, rule_id = expected_for(tensor, rules)
        actual = artifact.get(tensor["name"])
        item = {
            "name": tensor["name"],
            "class": tensor["class"],
            "layer": tensor.get("layer"),
            "source_present": True,
            "present": actual is not None,
            "qtype": actual["qtype"] if actual is not None else None,
            "shape": actual["shape"] if actual is not None else None,
            "required_qtype": expected_type,
            "required_shape": expected_shape,
            "manifest_rule": rule_id,
        }
        supported, reason = support_reason(
            item, expected_type, expected_shape, actual, tensor)
        item["execution_path_supported"] = supported
        item["reason"] = reason
        reports.append(item)
        if not supported:
            failures.append(item)
    unexpected = sorted(artifact_names - required_names)
    if unexpected:
        failures.extend({
            "name": name,
            "present": True,
            "qtype": artifact[name]["qtype"],
            "shape": artifact[name]["shape"],
            "execution_path_supported": False,
            "reason": "unexpected tensor is not part of the text runtime artifact",
        } for name in unexpected)
    report = {
        "format": "q38_runtime_preflight_v1",
        "checkpoint": str(args.model_dir.resolve()),
        "checkpoint_shards": shard_count,
        "artifact": str(args.artifact.resolve()),
        "artifact_metadata": {
            key: metadata.get(key)
            for key in ("general.architecture", "q38.runtime_only",
                        "q38.max_layer", "q38.quantized",
                        "q38.excluded_vision", "q38.excluded_mtp")
        },
        "required_tensor_count": len(required),
        "artifact_tensor_count": len(artifact),
        "missing_count": sum(not item["present"] for item in reports),
        "unsupported_count": len(failures),
        "status": "pass" if not failures else "fail",
        "tensors": reports,
        "unexpected_tensors": unexpected,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--classes", type=Path,
                        default=Path("artifacts/m1/tensor_classes.json"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("tools/quant_manifest_q2.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, PreflightError, KeyError, ValueError) as exc:
        report = {
            "format": "q38_runtime_preflight_v1",
            "checkpoint": str(args.model_dir.resolve()),
            "artifact": str(args.artifact.resolve()),
            "status": "fail",
            "required_tensor_count": None,
            "artifact_tensor_count": None,
            "missing_count": None,
            "unsupported_count": 1,
            "tensors": [],
            "unexpected_tensors": [],
            "failures": [{"execution_path_supported": False,
                          "reason": str(exc)}],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        print(f"q38 preflight failed: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
