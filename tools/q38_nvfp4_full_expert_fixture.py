#!/usr/bin/env python3
"""Extract compact full-expert NVFP4 fixtures and freeze an FP32 oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "reference"))
import nvfp4_modelopt_reference as ref

from q38_nvfp4_pack import (  # noqa: E402
    COMPONENT_ID,
    PROJECTION_ID,
    classify,
    collect_tensors,
    source_files,
)


MAGIC = b"NVF4FULL"
PROJECTIONS = ("gate", "up", "down")
STAGES = {"early": 0, "middle": 1, "late": 2}
LAYER_FOR_STAGE = {"early": 0, "middle": 24, "late": 47}


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_tensor(source_root: Path, tensor) -> bytes:
    path = source_root / tensor.source_name
    with path.open("rb", buffering=0) as fp:
        fp.seek(tensor.source_offset)
        payload = fp.read(tensor.bytes)
    if len(payload) != tensor.bytes:
        raise RuntimeError(f"short read for {tensor.name}")
    return payload


def decode_weight_matvec(
    packed_weight: bytes,
    weight_scale: bytes,
    weight_scale_2: float,
    packed_activation: bytes,
    activation_scale: bytes,
    input_scale: float,
    rows: int,
    logical_k: int,
) -> list[float]:
    result = []
    for row in range(rows):
        accumulator = 0.0
        for logical_k_index in range(logical_k):
            weight_byte = packed_weight[row * (logical_k // 2) + logical_k_index // 2]
            activation_byte = packed_activation[logical_k_index // 2]
            weight_code = (
                weight_byte >> 4 if logical_k_index & 1 else weight_byte & 0xF
            )
            activation_code = (
                activation_byte >> 4
                if logical_k_index & 1
                else activation_byte & 0xF
            )
            weight_value = ref.decode_fp4_nibble(weight_code)
            activation_value = ref.decode_fp4_nibble(activation_code)
            weight_scale_value = (
                ref.decode_e4m3fn_byte(
                    weight_scale[row * (logical_k // 16) + logical_k_index // 16]
                )
                * weight_scale_2
            )
            activation_scale_value = (
                ref.decode_e4m3fn_byte(activation_scale[logical_k_index // 16])
                * input_scale
            )
            product = f32(
                weight_value
                * weight_scale_value
                * activation_value
                * activation_scale_value
            )
            accumulator = f32(accumulator + product)
        result.append(accumulator)
    return result


def extract(source_root: Path, stage: str, output_dir: Path) -> None:
    sources = source_files(source_root)
    tensors = collect_tensors(source_root, sources)
    experts, _, _, _, _, _, _, _ = classify(tensors)
    layer = LAYER_FOR_STAGE[stage]
    expert_id = 0

    hidden_path = Path("tests/fixtures/moe") / stage / "hidden.f32"
    activation_payload = hidden_path.read_bytes()
    if len(activation_payload) != 2560 * 4:
        raise RuntimeError(f"unexpected activation fixture size: {hidden_path}")
    activation = list(struct.unpack("<2560f", activation_payload))

    projection_data = {}
    for projection in PROJECTIONS:
        projection_data[projection] = {}
        projection_id = PROJECTION_ID[projection]
        rows = 2560 if projection == "down" else 640
        logical_k = 640 if projection == "down" else 2560
        for component in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            tensor = experts[
                (layer, expert_id, projection_id, COMPONENT_ID[component])
            ].tensor
            payload = read_tensor(source_root, tensor)
            projection_data[projection][component] = {
                "tensor": tensor,
                "payload": payload,
                "rows": rows,
                "logical_k": logical_k,
            }

    oracle = {}
    for projection in ("gate", "up"):
        input_scale_payload = projection_data[projection]["input_scale"]["payload"]
        input_scale = struct.unpack("<f", input_scale_payload)[0]
        quantized = ref.quantize_activation(activation, input_scale)
        weight_scale_2 = struct.unpack(
            "<f", projection_data[projection]["weight_scale_2"]["payload"]
        )[0]
        data = projection_data[projection]
        output = decode_weight_matvec(
            data["weight"]["payload"],
            data["weight_scale"]["payload"],
            weight_scale_2,
            quantized["packed"],
            quantized["block_scale_bytes"],
            input_scale,
            data["weight"]["rows"],
            data["weight"]["logical_k"],
        )
        oracle[projection] = {
            "input": activation,
            "quantized": quantized,
            "output": output,
        }

    intermediate = [
        f32(
            gate / (1.0 + __import__("math").exp(-gate)) * up
        )
        for gate, up in zip(oracle["gate"]["output"], oracle["up"]["output"])
    ]
    down_input_scale = struct.unpack(
        "<f", projection_data["down"]["input_scale"]["payload"]
    )[0]
    down_quantized = ref.quantize_activation(intermediate, down_input_scale)
    down_data = projection_data["down"]
    down_scale_2 = struct.unpack(
        "<f", down_data["weight_scale_2"]["payload"]
    )[0]
    down_output = decode_weight_matvec(
        down_data["weight"]["payload"],
        down_data["weight_scale"]["payload"],
        down_scale_2,
        down_quantized["packed"],
        down_quantized["block_scale_bytes"],
        down_input_scale,
        down_data["weight"]["rows"],
        down_data["weight"]["logical_k"],
    )
    oracle["down"] = {
        "input": intermediate,
        "quantized": down_quantized,
        "output": down_output,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "payload.bin"
    with payload_path.open("wb") as fp:
        fp.write(struct.pack(
            "<8sIIII", MAGIC, 1, STAGES[stage], layer, expert_id
        ))
        fp.write(struct.pack("<I", len(PROJECTIONS)))
        for projection in PROJECTIONS:
            data = projection_data[projection]
            tensor = data["weight"]["tensor"]
            quantized = oracle[projection]["quantized"]
            input_values = oracle[projection]["input"]
            output = oracle[projection]["output"]
            fp.write(struct.pack(
                "<8Iff",
                PROJECTION_ID[projection],
                data["weight"]["rows"],
                data["weight"]["logical_k"],
                len(data["weight"]["payload"]),
                len(data["weight_scale"]["payload"]),
                len(input_values),
                len(quantized["packed"]),
                len(quantized["block_scale_bytes"]),
                struct.unpack(
                    "<f", data["weight_scale_2"]["payload"]
                )[0],
                struct.unpack("<f", data["input_scale"]["payload"])[0],
            ))
            fp.write(data["weight"]["payload"])
            fp.write(data["weight_scale"]["payload"])
            fp.write(struct.pack(f"<{len(input_values)}f", *input_values))
            fp.write(quantized["packed"])
            fp.write(quantized["block_scale_bytes"])
            fp.write(struct.pack(f"<{len(output)}f", *output))
        fp.write(struct.pack("<640f", *intermediate))

    tensor_metadata = {}
    for projection in PROJECTIONS:
        tensor_metadata[projection] = {}
        for component in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            tensor = projection_data[projection][component]["tensor"]
            payload = projection_data[projection][component]["payload"]
            tensor_metadata[projection][component] = {
                "name": tensor.name,
                "source_shard": tensor.source_name,
                "source_offset": tensor.source_offset,
                "bytes": tensor.bytes,
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "payload_sha256": sha256_bytes(payload),
            }
    metadata = {
        "format": "q38-m9n-04-full-expert-fixture-v1",
        "stage": stage,
        "layer": layer,
        "expert_id": expert_id,
        "checkpoint_revision": "fc694b54fb0174e0913e6adf86691ef85a4ead47",
        "source_model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
        "source_activation": {
            "path": str(hidden_path),
            "dtype": "F32",
            "shape": [2560],
            "sha256": sha256_bytes(activation_payload),
        },
        "tensors": tensor_metadata,
        "payload": {
            "path": "payload.bin",
            "bytes": payload_path.stat().st_size,
            "sha256": sha256_bytes(payload_path.read_bytes()),
        },
        "oracle": {
            "weight_formula": "decode_e2m1 * decode_e4m3fn(weight_scale) * weight_scale_2",
            "activation_block_size": 16,
            "accumulation": "FP32 rounded after each product and sum",
            "intermediate": "FP32 SiLU(gate) * up",
        },
        "expected": {
            "gate": {
                "output_shape": [640],
                "output_sha256": sha256_bytes(
                    struct.pack("<640f", *oracle["gate"]["output"])
                ),
            },
            "up": {
                "output_shape": [640],
                "output_sha256": sha256_bytes(
                    struct.pack("<640f", *oracle["up"]["output"])
                ),
            },
            "intermediate": {
                "output_shape": [640],
                "output_sha256": sha256_bytes(
                    struct.pack("<640f", *intermediate)
                ),
            },
            "down": {
                "output_shape": [2560],
                "output_sha256": sha256_bytes(
                    struct.pack("<2560f", *oracle["down"]["output"])
                ),
            },
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    extract(args.source_root, args.stage, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
