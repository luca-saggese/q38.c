"""Minimal, dependency-free reference for the ModelOpt 0.46 NVFP4 ABI.

The formulas mirror the pinned ModelOpt sources:

* modelopt/torch/quantization/qtensor/nvfp4_tensor.py
* modelopt/torch/kernels/quantization/common/nvfp4_quant.py
* modelopt/torch/kernels/quantization/gemm/fp4_kernel.py
* modelopt/torch/kernels/quantization/gemm/fp4_kernel_hopper.py

This module intentionally operates on small byte slices. It does not load a
checkpoint and does not depend on CUDA, PyTorch, or ModelOpt at runtime.
"""

from __future__ import annotations

import math
import struct
from typing import Iterable, Sequence

E2M1_MAX = 6.0
E4M3_MAX = 448.0
NVFP4_BLOCK_SIZE = 16

E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

E2M1_CODE_TABLE = tuple(
    {
        "nibble": nibble,
        "sign_bit": (nibble >> 3) & 1,
        "magnitude_code": nibble & 0x7,
        "value": E2M1_VALUES[nibble],
    }
    for nibble in range(16)
)


def decode_fp4_nibble(nibble: int) -> float:
    """Decode one packed NVFP4 E2M1 nibble."""
    if not 0 <= nibble <= 0xF:
        raise ValueError(f"FP4 nibble out of range: {nibble}")
    return E2M1_VALUES[nibble]


def unpack_fp4(packed: bytes | bytearray | Sequence[int], logical_count: int | None = None) -> list[float]:
    """Unpack low-nibble-first FP4 values from U8 storage."""
    values: list[float] = []
    for byte in packed:
        values.append(decode_fp4_nibble(byte & 0x0F))
        values.append(decode_fp4_nibble((byte >> 4) & 0x0F))
    if logical_count is not None:
        if logical_count < 0 or logical_count > len(values):
            raise ValueError("logical_count exceeds packed payload")
        return values[:logical_count]
    return values


def decode_e4m3fn_byte(byte: int) -> float:
    """Decode one finite-only FP8 E4M3 byte used by ModelOpt scales."""
    if not 0 <= byte <= 0xFF:
        raise ValueError(f"FP8 byte out of range: {byte}")
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0x0F
    mantissa = byte & 0x07

    if exponent == 0:
        value = (mantissa / 8.0) * 2.0**-6
    elif exponent == 0x0F:
        if mantissa == 0x07:
            return math.nan
        value = (1.0 + mantissa / 8.0) * 2.0**8
    else:
        value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return sign * value


def decode_e4m3fn(data: bytes | bytearray | Sequence[int]) -> list[float]:
    """Decode a sequence of finite-only FP8 E4M3 bytes."""
    return [decode_e4m3fn_byte(byte) for byte in data]


def _round_to_even(value: float) -> int:
    lower = math.floor(value)
    fraction = value - lower
    if fraction < 0.5:
        return lower
    if fraction > 0.5:
        return lower + 1
    return lower if lower % 2 == 0 else lower + 1


def encode_e4m3fn(value: float) -> int:
    """Encode a finite scalar with round-to-nearest-even E4M3FN semantics."""
    if math.isnan(value):
        return 0x7F
    sign = 0x80 if value < 0 else 0
    magnitude = abs(value)
    if magnitude == 0.0:
        return sign
    if math.isinf(magnitude) or magnitude >= E4M3_MAX:
        return sign | 0x7E

    min_normal = 2.0**-6
    subnormal_step = 2.0**-9
    if magnitude < min_normal:
        mantissa = _round_to_even(magnitude / subnormal_step)
        if mantissa >= 8:
            return sign | (1 << 3)
        return sign | mantissa

    exponent = math.floor(math.log2(magnitude))
    exponent_field = exponent + 7
    mantissa = _round_to_even((magnitude / 2.0**exponent - 1.0) * 8.0)
    if mantissa >= 8:
        exponent_field += 1
        mantissa = 0
    if exponent_field >= 0x0F:
        return sign | 0x7E
    return sign | (exponent_field << 3) | mantissa


def encode_fp4(value: float) -> int:
    """Encode one already-normalized value using ModelOpt FP4 decisions."""
    magnitude = abs(value)
    if magnitude <= 0.25:
        code = 0
    elif magnitude < 0.75:
        code = 1
    elif magnitude <= 1.25:
        code = 2
    elif magnitude < 1.75:
        code = 3
    elif magnitude <= 2.5:
        code = 4
    elif magnitude < 3.5:
        code = 5
    elif magnitude <= 5.0:
        code = 6
    else:
        code = 7
    return code | (0x8 if value < 0.0 else 0)


def dequantize_weight_slice(
    packed_rows: Sequence[bytes],
    scale_rows: Sequence[bytes],
    weight_scale_2: float,
    logical_k: int,
) -> list[list[float]]:
    """Reconstruct a small row-major NVFP4 weight slice."""
    if len(packed_rows) != len(scale_rows):
        raise ValueError("packed and scale row counts differ")
    if logical_k <= 0 or logical_k % NVFP4_BLOCK_SIZE:
        raise ValueError("logical_k must be a positive multiple of 16")
    blocks = logical_k // NVFP4_BLOCK_SIZE
    output: list[list[float]] = []
    for packed, scale_bytes in zip(packed_rows, scale_rows):
        values = unpack_fp4(packed, logical_k)
        scales = decode_e4m3fn(scale_bytes)
        if len(scales) != blocks:
            raise ValueError("scale row does not cover logical_k")
        output.append(
            [
                values[k] * scales[k // NVFP4_BLOCK_SIZE] * weight_scale_2
                for k in range(logical_k)
            ]
        )
    return output


def modelopt_authoritative_dequant(
    packed_rows: Sequence[bytes],
    scale_rows: Sequence[bytes],
    weight_scale_2: float,
    logical_k: int,
) -> list[list[float]]:
    """Direct translation of ModelOpt NVFP4QTensor.dequantize."""
    blocks = logical_k // NVFP4_BLOCK_SIZE
    output: list[list[float]] = []
    for packed, scale_bytes in zip(packed_rows, scale_rows):
        unpacked: list[float] = []
        for byte in packed:
            unpacked.append(E2M1_VALUES[byte & 0x0F])
            unpacked.append(E2M1_VALUES[(byte >> 4) & 0x0F])
        scales = [decode_e4m3fn_byte(byte) * weight_scale_2 for byte in scale_bytes]
        if len(scales) != blocks or len(unpacked) < logical_k:
            raise ValueError("slice dimensions do not match block scales")
        output.append([unpacked[k] * scales[k // NVFP4_BLOCK_SIZE] for k in range(logical_k)])
    return output


def quantize_activation(
    values: Sequence[float],
    input_scale: float,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> dict[str, object]:
    """Quantize activation values using ModelOpt's W4A4 scale convention.

    ``input_scale`` is the exported ``amax / (6 * 448)`` normalization scale.
    Each activation block gets a dynamically generated FP8 E4M3 scale.
    """
    if input_scale <= 0.0 or not math.isfinite(input_scale):
        raise ValueError("input_scale must be finite and positive")
    if block_size <= 0 or len(values) % block_size:
        raise ValueError("activation length must be divisible by block_size")

    packed = bytearray()
    block_scale_bytes: list[int] = []
    dequantized: list[float] = []
    for start in range(0, len(values), block_size):
        block = list(values[start : start + block_size])
        block_amax = max(abs(value) for value in block)
        raw_scale = block_amax / (E2M1_MAX * input_scale)
        scale_byte = encode_e4m3fn(raw_scale)
        block_scale_bytes.append(scale_byte)
        block_scale = decode_e4m3fn_byte(scale_byte) * input_scale
        if block_scale < 1.0e-5:
            block_scale = 1.0

        codes = [encode_fp4(value / block_scale) for value in block]
        for index in range(0, len(codes), 2):
            packed.append(codes[index] | (codes[index + 1] << 4))
            dequantized.extend(
                (
                    decode_fp4_nibble(codes[index]) * block_scale,
                    decode_fp4_nibble(codes[index + 1]) * block_scale,
                )
            )

    return {
        "packed": bytes(packed),
        "block_scale_bytes": bytes(block_scale_bytes),
        "block_scale_values": decode_e4m3fn(block_scale_bytes),
        "dequantized": dequantized,
        "runtime_global_scale": 1.0 / input_scale,
    }


def matvec(weight: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    """Compute a small deterministic reference matrix-vector product."""
    if any(len(row) != len(vector) for row in weight):
        raise ValueError("matvec dimensions do not match")
    return [sum(value * x for value, x in zip(row, vector)) for row in weight]


def f32(value: float) -> float:
    """Round a Python scalar to the checkpoint's little-endian F32 format."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def max_abs_difference(left: Iterable[float], right: Iterable[float]) -> float:
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)
