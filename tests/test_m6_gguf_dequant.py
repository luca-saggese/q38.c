#!/usr/bin/env python3
"""Fail-closed byte-level checks for the GGUF dequantization fixtures."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.m6_quant_matched_reference import GGUF


EXPECTED = {
    "model.language_model.embed_tokens.weight": (30, (248320, 2560), 9419),
    "model.language_model.layers.1.ple.key_proj.weight": (8, (10240, 2560), 0),
    "model.language_model.layers.2.mlp.experts.gate_up_proj": (
        10, (512, 1280, 2560), 75 * 1280),
}


def bf16(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def q2_oracle(raw: memoryview, cols: int) -> list[float]:
    result = [0.0] * cols
    for block_index in range(cols // 256):
        base = block_index * 84
        scales = raw[base:base + 16]
        q = raw[base + 16:base + 80]
        d = struct.unpack_from("<e", raw, base + 80)[0]
        dmin = struct.unpack_from("<e", raw, base + 82)[0]
        at = block_index * 256
        scale_index = 0
        for half in (0, 128):
            qbase = half // 4
            for j in range(4):
                shift = 2 * j
                for part, offset in enumerate((0, 16)):
                    scale = scales[scale_index]
                    scale_index += 1
                    dl = d * (scale & 15)
                    ml = dmin * (scale >> 4)
                    start = at + half + j * 32 + part * 16
                    for l in range(16):
                        result[start + l] = (
                            dl * ((q[qbase + offset + l] >> shift) & 3) - ml
                        )
    return result


def oracle(reader: GGUF, name: str, row: int) -> tuple[bytes, list[float]]:
    descriptor = reader.descriptor(name)
    shape = descriptor["shape"]
    rows, cols = math.prod(shape[:-1]), shape[-1]
    block, block_bytes = {8: (32, 34), 10: (256, 84)}.get(
        descriptor["type"], (1, 2)
    )
    row_bytes = cols // block * block_bytes
    start = descriptor["offset"] + row * row_bytes
    raw = memoryview(reader.mm)[start:start + row_bytes]
    if descriptor["type"] == 30:
        result = [
            bf16(struct.unpack_from("<H", raw, i * 2)[0])
            for i in range(cols)
        ]
    elif descriptor["type"] == 8:
        result = [
            struct.unpack_from("<e", raw, (i // 32) * 34)[0] *
            struct.unpack_from("<b", raw, (i // 32) * 34 + 2 + i % 32)[0]
            for i in range(cols)
        ]
    elif descriptor["type"] == 10:
        result = q2_oracle(raw, cols)
    else:
        raise AssertionError(f"unsupported fixture type: {descriptor['type']}")
    return bytes(raw), result


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_m6_gguf_dequant.py model.gguf fixture.json")
    model, fixture_path = Path(sys.argv[1]), Path(sys.argv[2])
    report = json.loads(fixture_path.read_text())
    if report.get("format") != "q38-m6-gguf-dequant-fixtures-v1":
        raise SystemExit("fixture format mismatch")
    reader = GGUF(model)
    try:
        fixtures = report.get("fixtures")
        if not isinstance(fixtures, list) or len(fixtures) != len(EXPECTED):
            raise SystemExit("fixture set is incomplete")
        for fixture in fixtures:
            name = fixture.get("tensor")
            if name not in EXPECTED:
                raise SystemExit(f"unexpected fixture tensor: {name}")
            expected_type, expected_shape, expected_row = EXPECTED[name]
            descriptor = reader.descriptor(name)
            if (fixture.get("type"), tuple(fixture.get("shape", [])),
                    fixture.get("row")) != (
                        expected_type, expected_shape, expected_row):
                raise SystemExit(f"{name}: type/shape/row mismatch")
            raw, expected = oracle(reader, name, expected_row)
            expected_stride = len(raw)
            if fixture.get("row_bytes") != expected_stride:
                raise SystemExit(f"{name}: row stride mismatch")
            if fixture.get("raw_sha256") is None:
                fixture["raw_sha256"] = hashlib.sha256(raw).hexdigest()
            elif fixture["raw_sha256"] != hashlib.sha256(raw).hexdigest():
                raise SystemExit(f"{name}: raw checksum mismatch")
            actual = fixture.get("values")
            if not isinstance(actual, list) or len(actual) != len(expected):
                raise SystemExit(f"{name}: decoded row length mismatch")
            if any(not math.isclose(float(a), b, rel_tol=1e-6, abs_tol=1e-6)
                   for a, b in zip(actual, expected)):
                raise SystemExit(f"{name}: q38/oracle decoded values diverge")
            if int(fixture.get("raw_fnv1a", "0"), 16) == 0:
                raise SystemExit(f"{name}: missing q38 raw checksum")
        fixture_path.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        reader.close()
    print("test_m6_gguf_dequant: BF16/Q8_0/Q2_K GGUF fixtures passed")


if __name__ == "__main__":
    main()
