#!/usr/bin/env python3
"""Small, model-free checks for the independent GGUF quant decoder."""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.m6_quant_matched_reference import GGUF


def main() -> None:
    block = bytearray(84)
    struct.pack_into("<H", block, 80, 0x3C00)  # d = 1
    for i in range(16):
        block[i] = 1
    for i in range(64):
        block[16 + i] = 0xE4
    decoded = GGUF._q2_row(memoryview(block), 256)
    assert len(decoded) == 256
    for i, value in enumerate(decoded):
        assert value == (i // 32) % 4, (i, value)
    print("test_m6_quant_reference: Q2_K direct GGUF row decode passed")


if __name__ == "__main__":
    main()
