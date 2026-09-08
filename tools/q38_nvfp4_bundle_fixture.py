#!/usr/bin/env python3
"""Build a compact resident-weight bundle for the M9N-05 routed benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from q38_nvfp4_full_expert_fixture import (
    COMPONENT_ID,
    PROJECTION_ID,
    LAYER_FOR_STAGE,
    read_tensor,
)
from q38_nvfp4_pack import classify, collect_tensors, source_files


MAGIC = b"NVF4BNDL"
PROJECTIONS = ("gate", "up", "down")
STAGES = {"early": 0, "middle": 1, "late": 2}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_f32_file(path: Path, count: int) -> tuple[float, ...]:
    payload = path.read_bytes()
    if len(payload) != count * 4:
        raise RuntimeError(f"unexpected size for {path}")
    return struct.unpack(f"<{count}f", payload)


def extract(source_root: Path, stage: str, output_dir: Path) -> None:
    source_entries = source_files(source_root)
    tensors = collect_tensors(source_root, source_entries)
    experts, _, _, _, _, _, _, _ = classify(tensors)

    fixture_dir = Path("tests/fixtures/moe") / stage
    metadata = json.loads((fixture_dir / "metadata.json").read_text())
    expert_ids = [int(value) for value in metadata["selected_experts"]]
    route_weights = read_f32_file(fixture_dir / "selected_weights.f32", 10)
    hidden = fixture_dir / "hidden.f32"
    hidden_payload = hidden.read_bytes()
    if len(hidden_payload) != 2560 * 4:
        raise RuntimeError(f"unexpected hidden fixture size: {hidden}")

    layer = LAYER_FOR_STAGE[stage]
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "payload.bin"
    tensor_metadata: list[dict[str, object]] = []

    with payload_path.open("wb") as output:
        output.write(struct.pack(
            "<8sIIIII",
            MAGIC,
            1,
            STAGES[stage],
            layer,
            10,
            2560,
        ))
        output.write(hidden_payload)
        for slot, (expert_id, route_weight) in enumerate(
            zip(expert_ids, route_weights)
        ):
            output.write(struct.pack("<If", expert_id, route_weight))
            expert_metadata: dict[str, object] = {
                "slot": slot,
                "expert_id": expert_id,
                "route_weight": route_weight,
                "tensors": {},
            }
            for projection in PROJECTIONS:
                projection_id = PROJECTION_ID[projection]
                rows = 2560 if projection == "down" else 640
                logical_k = 640 if projection == "down" else 2560
                values: dict[str, bytes] = {}
                component_metadata: dict[str, object] = {}
                for component in (
                    "weight",
                    "weight_scale",
                    "weight_scale_2",
                    "input_scale",
                ):
                    tensor = experts[
                        (
                            layer,
                            expert_id,
                            projection_id,
                            COMPONENT_ID[component],
                        )
                    ].tensor
                    data = read_tensor(source_root, tensor)
                    values[component] = data
                    component_metadata[component] = {
                        "name": tensor.name,
                        "source_shard": tensor.source_name,
                        "source_offset": tensor.source_offset,
                        "bytes": tensor.bytes,
                        "dtype": tensor.dtype,
                        "shape": list(tensor.shape),
                        "payload_sha256": sha256_bytes(data),
                    }
                weight_scale_2 = struct.unpack(
                    "<f", values["weight_scale_2"]
                )[0]
                input_scale = struct.unpack(
                    "<f", values["input_scale"]
                )[0]
                output.write(struct.pack(
                    "<8Iff",
                    projection_id,
                    rows,
                    logical_k,
                    len(values["weight"]),
                    len(values["weight_scale"]),
                    0,
                    0,
                    0,
                    weight_scale_2,
                    input_scale,
                ))
                output.write(values["weight"])
                output.write(values["weight_scale"])
                expert_metadata["tensors"][projection] = component_metadata
            tensor_metadata.append(expert_metadata)

    metadata_out = {
        "format": "q38-m9n-05-nvfp4-bundle-v1",
        "stage": stage,
        "layer": layer,
        "activation_capture_layer": metadata.get("layer"),
        "source_model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
        "checkpoint_revision": "fc694b54fb0174e0913e6adf86691ef85a4ead47",
        "expert_count": 10,
        "hidden_shape": [2560],
        "expert_ids": expert_ids,
        "route_weights": list(route_weights),
        "source_activation": {
            "path": str(hidden),
            "sha256": sha256_bytes(hidden_payload),
        },
        "tensors": tensor_metadata,
        "payload": {
            "path": "payload.bin",
            "bytes": payload_path.stat().st_size,
            "sha256": sha256_bytes(payload_path.read_bytes()),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata_out, indent=2) + "\n"
    )


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
