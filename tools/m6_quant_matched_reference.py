#!/usr/bin/env python3
"""Independent Qwen4Exp reference which reads the q38 GGUF directly.

The reference deliberately has no q38 import and never calls q38_forward.  It
maps GGUF tensor payloads, decodes BF16/Q8_0/Q2_K rows, and supplies the
dequantized routed experts to the official Qwen4Exp equations.  Only experts
selected for the current token are materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import struct
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

_qwen4 = __import__(
    "transformers.models.qwen4_exp.modeling_qwen4_exp",
    fromlist=["Qwen4ExpTextDecoderLayer", "Qwen4ExpTextGatedResidual",
              "Qwen4ExpTextRotaryEmbedding"],
)
Qwen4ExpTextDecoderLayer = _qwen4.Qwen4ExpTextDecoderLayer
Qwen4ExpTextGatedResidual = _qwen4.Qwen4ExpTextGatedResidual
Qwen4ExpTextRotaryEmbedding = _qwen4.Qwen4ExpTextRotaryEmbedding


FIXED = (0, 1, 2, 3)
GGUF_Q8_0 = 8
GGUF_Q2_K = 10
GGUF_BF16 = 30
GGUF_I64 = 27
TYPE_INFO = {
    GGUF_Q8_0: (32, 34),
    GGUF_Q2_K: (256, 84),
    GGUF_BF16: (1, 2),
    GGUF_I64: (1, 8),
}


def stats(values: torch.Tensor) -> dict:
    flat = values.detach().float().reshape(-1).cpu()
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    raw = flat.numpy().tobytes()
    if finite_values.numel():
        minimum, maximum = float(finite_values.min()), float(finite_values.max())
        mean = float(finite_values.mean())
        rms = float(torch.sqrt(torch.mean(finite_values * finite_values)))
        max_abs = float(torch.max(torch.abs(finite_values)))
        indices = torch.where(finite)[0]
        min_index = int(indices[torch.argmin(finite_values)])
        max_index = int(indices[torch.argmax(finite_values)])
        max_abs_index = int(indices[torch.argmax(torch.abs(finite_values))])
        extreme_indices = sorted(
            range(flat.numel()), key=lambda i: abs(float(flat[i])),
            reverse=True
        )[:4]
        extreme_coordinates = [
            {"index": i, "value": float(flat[i])} for i in extreme_indices
        ]
    else:
        minimum = maximum = mean = rms = max_abs = None
        min_index = max_index = max_abs_index = None
        extreme_coordinates = []
    return {
        "min": minimum, "max": maximum, "mean": mean, "rms": rms,
        "max_abs": max_abs, "finite_count": int(finite.sum()),
        "min_index": min_index, "max_index": max_index,
        "max_abs_index": max_abs_index, "nan_count": int(torch.isnan(flat).sum()),
        "inf_count": int(torch.isinf(flat).sum()),
        "checksum": hashlib.sha256(raw).hexdigest(),
        "extreme_coordinates": extreme_coordinates,
        "fixed": [{"index": i, "value": float(flat[i]) if i < flat.numel()
                   else None} for i in FIXED],
    }


def values(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().float().reshape(-1).cpu().tolist()


def bf16_bits(tensor: torch.Tensor) -> list[int]:
    raw = tensor.detach().float().reshape(-1).cpu().view(torch.int32)
    return ((raw.to(torch.int64) & 0xffffffff) >> 16).tolist()


class GGUF:
    """Small read-only GGUF v3 reader; payloads remain mmap-backed."""

    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.tensors: dict[str, dict] = {}
        self.metadata: dict[str, object] = {}
        self.profile = None
        magic, version, tensor_count, kv_count = struct.unpack_from(
            "<IIQQ", self.mm, 0)
        if magic != 0x46554747 or version != 3:
            raise ValueError("expected GGUF v3")
        pos = 24
        for _ in range(kv_count):
            key, pos = self._string(pos)
            kind = struct.unpack_from("<I", self.mm, pos)[0]
            pos += 4
            value, pos = self._value(pos, kind)
            self.metadata[key] = value
        self.data_start_header = pos
        alignment = int(self.metadata.get("general.alignment", 32))
        descriptors = []
        for _ in range(tensor_count):
            name, pos = self._string(pos)
            ndim = struct.unpack_from("<I", self.mm, pos)[0]
            pos += 4
            shape = struct.unpack_from("<" + "Q" * ndim, self.mm, pos)
            pos += 8 * ndim
            kind, relative = struct.unpack_from("<IQ", self.mm, pos)
            pos += 12
            if kind not in TYPE_INFO:
                raise ValueError(f"{name}: unsupported GGUF type {kind}")
            block, block_bytes = TYPE_INFO[kind]
            elements = math.prod(shape)
            byte_count = ((elements + block - 1) // block) * block_bytes
            descriptors.append((name, shape, kind, relative, byte_count))
        data_start = (pos + alignment - 1) // alignment * alignment
        for name, shape, kind, relative, byte_count in descriptors:
            absolute = data_start + relative
            if absolute + byte_count > len(self.mm):
                raise ValueError(f"{name}: payload outside GGUF")
            self.tensors[name] = {
                "shape": tuple(int(x) for x in shape), "type": kind,
                "offset": absolute, "bytes": byte_count,
            }

    def _string(self, pos: int) -> tuple[str, int]:
        length = struct.unpack_from("<Q", self.mm, pos)[0]
        pos += 8
        end = pos + length
        return bytes(self.mm[pos:end]).decode("utf-8"), end

    def _value(self, pos: int, kind: int) -> tuple[object, int]:
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                 10: 8, 11: 8, 12: 8}
        if kind == 8:
            return self._string(pos)
        if kind in sizes:
            size = sizes[kind]
            raw = bytes(self.mm[pos:pos + size])
            formats = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I",
                       5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q",
                       12: "<d"}
            return struct.unpack(formats[kind], raw)[0], pos + size
        if kind == 9:
            item_kind = struct.unpack_from("<I", self.mm, pos)[0]
            count = struct.unpack_from("<Q", self.mm, pos + 4)[0]
            pos += 12
            result = []
            for _ in range(count):
                item, pos = self._value(pos, item_kind)
                result.append(item)
            return result, pos
        raise ValueError(f"unsupported GGUF metadata type {kind}")

    def close(self) -> None:
        self.mm.close()
        self.file.close()

    def set_profile(self, profile: object | None) -> None:
        """Attach a timing sink used by the stateful oracle."""
        self.profile = profile

    def _record(self, field: str, elapsed: float) -> None:
        if self.profile is not None:
            setattr(
                self.profile,
                field,
                getattr(self.profile, field, 0.0) + elapsed * 1000.0,
            )

    def _to_device(self, value: torch.Tensor, device: torch.device) -> torch.Tensor:
        if value.device == device:
            return value
        start = time.perf_counter()
        moved = value.to(device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._record("h2d_ms", time.perf_counter() - start)
        return moved

    def descriptor(self, name: str) -> dict:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"missing GGUF tensor {name}") from exc

    def raw_rows(self, name: str, row_indices: Iterable[int]) -> list[bytes]:
        d = self.descriptor(name)
        rows = math.prod(d["shape"][:-1])
        row_bytes = d["bytes"] // rows
        result = []
        for row in row_indices:
            if row < 0 or row >= rows:
                raise IndexError(f"{name}: row {row} outside tensor")
            start = d["offset"] + row * row_bytes
            result.append(bytes(self.mm[start:start + row_bytes]))
        return result

    def dense(self, name: str, device: torch.device) -> torch.Tensor:
        d = self.descriptor(name)
        if d["type"] == GGUF_BF16:
            start = time.perf_counter()
            count = math.prod(d["shape"])
            raw = memoryview(self.mm)[d["offset"]:d["offset"] + d["bytes"]]
            value = torch.frombuffer(raw, dtype=torch.uint16, count=count).view(
                torch.bfloat16).reshape(d["shape"]).float().clone()
            self._record("gguf_read_ms", time.perf_counter() - start)
            return self._to_device(value, device)
        if d["type"] == GGUF_I64:
            start = time.perf_counter()
            raw = memoryview(self.mm)[d["offset"]:d["offset"] + d["bytes"]]
            value = torch.frombuffer(raw, dtype=torch.int64,
                                     count=math.prod(d["shape"])).reshape(
                                         d["shape"]).clone()
            self._record("gguf_read_ms", time.perf_counter() - start)
            return self._to_device(value, device)
        if d["type"] == GGUF_Q8_0:
            rows, cols = math.prod(d["shape"][:-1]), d["shape"][-1]
            return self.quant_rows(name, range(rows), device).reshape(
                d["shape"])
        raise ValueError(f"{name}: dense decode not supported for type {d['type']}")

    def dense_rows(self, name: str, rows: Iterable[int],
                   device: torch.device) -> torch.Tensor:
        d = self.descriptor(name)
        if len(d["shape"]) != 2:
            raise ValueError(f"{name}: row slice requires a matrix")
        if d["type"] == GGUF_BF16:
            cols = d["shape"][1]
            result = []
            for row in rows:
                read_start = time.perf_counter()
                offset = d["offset"] + row * cols * 2
                raw = memoryview(self.mm)[offset:offset + cols * 2]
                result.append(torch.frombuffer(raw, dtype=torch.uint16,
                                               count=cols).view(
                                                   torch.bfloat16).float())
                self._record("gguf_read_ms", time.perf_counter() - read_start)
            return self._to_device(torch.stack(result), device)
        return self.quant_rows(name, rows, device)

    def quant_rows(self, name: str, row_indices: Iterable[int],
                   device: torch.device) -> torch.Tensor:
        d = self.descriptor(name)
        shape = d["shape"]
        if len(shape) < 2:
            raise ValueError(f"{name}: row decode requires a matrix")
        rows, cols = math.prod(shape[:-1]), shape[-1]
        block, block_bytes = TYPE_INFO[d["type"]]
        if cols % block:
            raise ValueError(f"{name}: row is not block aligned")
        row_bytes = (cols // block) * block_bytes
        raw = memoryview(self.mm)
        decoded = []
        for row in row_indices:
            if row < 0 or row >= rows:
                raise IndexError(f"{name}: row {row} outside tensor")
            read_start = time.perf_counter()
            start = d["offset"] + row * row_bytes
            row_data = raw[start:start + row_bytes]
            self._record("gguf_read_ms", time.perf_counter() - read_start)
            dequant_start = time.perf_counter()
            if d["type"] == GGUF_Q2_K:
                decoded.append(self._q2_row(row_data, cols))
            elif d["type"] == GGUF_Q8_0:
                decoded.append(self._q8_row(row_data, cols))
            else:
                raise ValueError(f"{name}: unsupported quantized type {d['type']}")
            self._record("dequant_ms", time.perf_counter() - dequant_start)
        result = torch.tensor(decoded, dtype=torch.float32)
        return self._to_device(result, device)

    @staticmethod
    def _q2_row(raw: memoryview, cols: int) -> list[float]:
        out = [0.0] * cols
        for block_index in range(cols // 256):
            base = block_index * 84
            scales = raw[base:base + 16]
            q = raw[base + 16:base + 80]
            d = struct.unpack_from("<e", raw, base + 80)[0]
            dmin = struct.unpack_from("<e", raw, base + 82)[0]
            out_base = block_index * 256
            scale_index = 0
            for half in (0, 128):
                qbase = half // 4
                for j in range(4):
                    shift = 2 * j
                    scale = scales[scale_index]
                    scale_index += 1
                    dl = d * (scale & 0xF)
                    ml = dmin * (scale >> 4)
                    at = out_base + half + j * 32
                    for l in range(16):
                        out[at + l] = dl * ((q[qbase + l] >> shift) & 3) - ml
                    scale = scales[scale_index]
                    scale_index += 1
                    dl = d * (scale & 0xF)
                    ml = dmin * (scale >> 4)
                    for l in range(16):
                        out[at + 16 + l] = (
                            dl * ((q[qbase + 16 + l] >> shift) & 3) - ml)
        return out

    @staticmethod
    def _q8_row(raw: memoryview, cols: int) -> list[float]:
        out = [0.0] * cols
        for block in range(cols // 32):
            base = block * 34
            scale = struct.unpack_from("<e", raw, base)[0]
            for i in range(32):
                out[block * 32 + i] = scale * struct.unpack_from(
                    "<b", raw, base + 2 + i)[0]
        return out


class GGUFEmbedding(nn.Module):
    def __init__(self, reader: GGUF, names: list[str], device: torch.device):
        super().__init__()
        self.reader, self.names, self.device = reader, names, device
        self.starts = []
        total = 0
        for name in names:
            self.starts.append(total)
            total += reader.descriptor(name)["shape"][0]

    @property
    def weight(self) -> torch.Tensor:
        return torch.empty(0, device=self.device)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        output = []
        for value in ids.reshape(-1).tolist():
            shard = 0
            while shard + 1 < len(self.starts) and self.starts[shard + 1] <= value:
                shard += 1
            row = value - self.starts[shard]
            output.append(self.reader.quant_rows(
                self.names[shard], [row], self.device)[0])
        return torch.stack(output).reshape(*ids.shape, 160)


class RoutedExpertCache:
    """Bounded CUDA cache for dequantized routed expert matrices."""

    def __init__(self, reader: GGUF, device: torch.device, capacity: int):
        if capacity < 1:
            raise ValueError("expert cache capacity must be positive")
        self.reader = reader
        self.device = device
        self.capacity = capacity
        self.entries: OrderedDict[tuple[str, int], torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, name: str, expert: int, rows: int) -> torch.Tensor:
        key = (name, expert)
        value = self.entries.pop(key, None)
        if value is not None:
            self.hits += 1
            self.entries[key] = value
            return value
        self.misses += 1
        value = self.reader.quant_rows(
            name, range(expert * rows, (expert + 1) * rows), self.device
        ).reshape(rows, -1)
        self.entries[key] = value
        while len(self.entries) > self.capacity:
            _, evicted = self.entries.popitem(last=False)
            del evicted
            self.evictions += 1
        return value

    def evidence(self) -> dict:
        return {
            "capacity": self.capacity,
            "entries": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "devices": sorted({str(value.device) for value in self.entries.values()}),
        }


class QuantExperts(nn.Module):
    def __init__(self, reader: GGUF, gate_up: str, down: str,
                 intermediate: int, device: torch.device,
                 expert_cache: RoutedExpertCache | None = None):
        super().__init__()
        self.reader = reader
        self.gate_up_name, self.down_name = gate_up, down
        self.intermediate_dim, self.device = intermediate, device
        self.num_experts = reader.descriptor(gate_up)["shape"][0]
        self.act_fn = nn.SiLU()
        self.expert_cache = expert_cache

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        indices = top_k_index.reshape(-1, top_k_index.shape[-1])
        weights = top_k_weights.reshape(-1, top_k_weights.shape[-1])
        result = torch.zeros_like(hidden)
        for expert in torch.unique(indices).tolist():
            positions = torch.where(indices == expert)
            token_rows, slots = positions
            current = hidden[token_rows].float()
            gate_rows = self.reader.descriptor(self.gate_up_name)["shape"][1]
            if self.expert_cache is None:
                gate_up = self.reader.quant_rows(
                    self.gate_up_name,
                    range(expert * gate_rows, (expert + 1) * gate_rows),
                    self.device).reshape(gate_rows, -1)
            else:
                gate_up = self.expert_cache.get(
                    self.gate_up_name, int(expert), gate_rows
                )
            gate, up = F.linear(current, gate_up).chunk(2, dim=-1)
            current = self.act_fn(gate) * up
            down_rows = self.reader.descriptor(self.down_name)["shape"][1]
            if self.expert_cache is None:
                down = self.reader.quant_rows(
                    self.down_name,
                    range(expert * down_rows, (expert + 1) * down_rows),
                    self.device).reshape(down_rows, -1).transpose(0, 1)
            else:
                down = self.expert_cache.get(
                    self.down_name, int(expert), down_rows
                ).transpose(0, 1)
            current = F.linear(current, down)
            current = current * weights[token_rows, slots, None]
            result.index_add_(0, token_rows, current.to(result.dtype))
        return result.reshape_as(hidden_states)


class QuantRouter(nn.Module):
    """Native q38-compatible router: BF16 effective logits and pre-cast IDs."""
    def __init__(self, weight: torch.Tensor, top_k: int, device: torch.device,
                 diagnostics: bool = False):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.top_k, self.device, self.diagnostics = top_k, device, diagnostics

    def forward(self, hidden_states: torch.Tensor):
        x = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
        pre_cast = F.linear(x, self.weight.float())
        effective = pre_cast.to(torch.bfloat16)
        if self.diagnostics:
            self.last_fp32_reduction = torch.sum(
                x[:, None, :] * self.weight.float()[None, :, :], dim=-1
            ).detach()
            self.last_bf16_matmul = F.linear(
                x.to(torch.bfloat16), self.weight.to(torch.bfloat16)
            ).detach()
        probs_pre = torch.softmax(pre_cast, dim=-1)
        order = torch.argsort(probs_pre, dim=-1, descending=True, stable=True)
        selected = order[:, :self.top_k]
        selected_pre = probs_pre.gather(1, selected)
        selected_pre = selected_pre / selected_pre.sum(dim=-1, keepdim=True)
        probs_effective = torch.softmax(effective.float(), dim=-1)
        scores = probs_effective.gather(1, selected)
        scores = (scores / scores.sum(dim=-1, keepdim=True)).to(effective.dtype)
        self.last_pre_cast = pre_cast.detach()
        self.last_effective = effective.detach()
        self.last_probs_pre = probs_pre.detach()
        self.last_probs_effective = probs_effective.detach()
        self.last_selected = selected.detach()
        return effective.reshape(*hidden_states.shape[:-1], -1), scores.reshape(
            *hidden_states.shape[:-1], -1), selected.reshape(
                *hidden_states.shape[:-1], -1)


def replace_parameter(module: nn.Module, name: str, value: torch.Tensor) -> None:
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], nn.Parameter(value))


def replace_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def materialize_layer(reader: GGUF, config, index: int,
                      device: torch.device,
                      expert_cache: RoutedExpertCache | None = None) -> nn.Module:
    with torch.device("meta"):
        layer = Qwen4ExpTextDecoderLayer(config, index)
    prefix = f"model.language_model.layers.{index}."
    for name, _ in list(layer.named_parameters()):
        if name.startswith("mlp.experts."):
            continue
        if "ple_embedding.ngram_embedding.weight" in name:
            continue
        replace_parameter(layer, name, reader.dense(prefix + name, device))
    for name, _ in list(layer.named_buffers()):
        full_name = prefix + name
        if full_name in reader.tensors:
            replace_buffer(layer, name, reader.dense(full_name, device))
    gate_name = prefix + "mlp.gate.weight"
    layer.mlp.gate = QuantRouter(
        reader.dense(gate_name, device), config.num_experts_per_tok, device,
        diagnostics=index == 9)
    layer.mlp.experts = QuantExperts(
        reader, prefix + "mlp.experts.gate_up_proj",
        prefix + "mlp.experts.down_proj", config.moe_intermediate_size, device,
        expert_cache)
    if layer.ple is not None:
        names = sorted(
            [name for name in reader.tensors
             if name.startswith(prefix + "ple.ple_embedding.ngram_embedding.shard_")],
            key=lambda name: int(name.rsplit("shard_", 1)[1].split(".", 1)[0]))
        layer.ple.ple_embedding.ngram_embedding = GGUFEmbedding(
            reader, names, device)
    return layer.to(device)


def top_k(values_: torch.Tensor, count: int = 10) -> list[dict]:
    flat = values_.detach().float().reshape(-1).cpu().tolist()
    order = sorted(range(len(flat)), key=lambda i: (-flat[i], i))[:count]
    return [{"id": i, "value": flat[i]} for i in order]


def vector_metrics(left: list[float] | None, right: list[float] | None) -> dict:
    if left is None or right is None:
        return {"status": "missing"}
    if len(left) != len(right):
        return {"status": "fail", "reason": "length"}
    error = [a - b for a, b in zip(left, right)]
    rms = math.sqrt(sum(x * x for x in error) / len(error)) if error else 0.0
    right_rms = math.sqrt(sum(x * x for x in right) / len(right)) if right else 0.0
    left_norm = math.sqrt(sum(x * x for x in left))
    right_norm = math.sqrt(sum(x * x for x in right))
    return {
        "status": "pass",
        "max_abs": max((abs(x) for x in error), default=0.0),
        "rms": rms,
        "relative_rms": rms / right_rms if right_rms else None,
        "cosine_similarity": (sum(a * b for a, b in zip(left, right)) /
                              (left_norm * right_norm)
                              if left_norm and right_norm else None),
    }


def first_output(output: object) -> torch.Tensor:
    """Return the activation tensor from tensor-or-tuple module outputs."""
    if isinstance(output, tuple):
        return output[0]
    return output


def boundary_entry(name: str, tensor: torch.Tensor) -> dict:
    return {
        "name": name,
        "width": tensor.shape[-1],
        "elements": tensor.numel(),
        "values": values(tensor),
        "stats": stats(tensor),
    }


def native_router(native: dict, layer: int) -> list[float] | None:
    for item in native.get("router_logits", []):
        if item.get("layer") == layer:
            if "logits" in item:
                return item["logits"]
            top = item.get("top", [])
            if len(top) == 512:
                ordered = [0.0] * 512
                for value in top:
                    ordered[value["id"]] = value["value"]
                return ordered
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/home/lvx/q38model"))
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=str,
                        help="comma-separated token IDs; overrides trace tokens")
    parser.add_argument("--start-layer", type=int, default=3)
    parser.add_argument("--max-layer", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --device cpu")
    device = torch.device(args.device)
    native = json.loads(args.trace.read_text())
    tokens = ([int(value) for value in args.tokens.split(",") if value.strip()]
              if args.tokens is not None else native.get("tokens"))
    if not tokens:
        raise SystemExit("reference requires at least one token")
    raw_config = json.loads((args.model_dir / "config.json").read_text())
    config = _qwen4.Qwen4ExpTextConfig(**raw_config["text_config"])
    reader = GGUF(args.gguf)
    stages, routing = [], []
    boundary_sets = {}
    layer2_moe_trace = None
    qsa_selection = []
    embedding_tokens = tokens if args.tokens is not None else [tokens[0]]
    hidden = reader.dense_rows(
        "model.language_model.embed_tokens.weight", embedding_tokens, device
    ).reshape(1, len(embedding_tokens), config.hidden_size)
    hidden = hidden.repeat(1, 1, config.hc_count)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    rotary = Qwen4ExpTextRotaryEmbedding(config).to(device)
    if args.tokens is not None:
        positions = torch.arange(len(tokens), dtype=torch.long, device=device)
        position_ids = positions.view(1, 1, -1).expand(3, -1, -1)
    else:
        position_ids = torch.zeros((3, 1, len(tokens)), dtype=torch.long,
                                   device=device)
    position_embeddings = rotary(hidden, position_ids)
    full_mask = torch.ones((1, 1, len(tokens), len(tokens)), dtype=torch.bool,
                           device=device)
    conv_mask = torch.ones((1, len(tokens)), dtype=torch.bool, device=device)
    first_divergence = None
    with torch.no_grad():
        for layer_index in range(config.num_hidden_layers):
            if args.max_layer is not None and layer_index > args.max_layer:
                break
            print(f"quant reference layer {layer_index}", flush=True)
            layer = materialize_layer(reader, config, layer_index, device)
            captured = {}
            hook_handles = []
            hook_handles.append(layer.mlp.gate.register_forward_hook(
                lambda _m, _i, output: captured.__setitem__("router", output)))
            hook_handles.append(layer.mlp.register_forward_pre_hook(
                lambda _m, inputs: captured.__setitem__(
                    "moe_input", inputs[0].detach())))
            if layer_index in (1, 9):
                hook_handles.append(layer.register_forward_pre_hook(
                    lambda _m, inputs: captured.__setitem__(
                        "layer_input_pre_ple", inputs[0].detach())))
                if layer.ple is not None:
                    hook_handles.append(layer.ple.register_forward_hook(
                        lambda _m, _i, output: captured.__setitem__(
                            "ple_contribution", output.detach())))
                    hook_handles.append(layer.ple.ple_embedding.register_forward_hook(
                        lambda _m, _i, output: captured.__setitem__(
                            "ple_embedding", output.detach())))
                    hook_handles.append(layer.ple.key_proj.register_forward_hook(
                        lambda _m, _i, output: captured.__setitem__(
                            "ple_key_projection", output.detach())))
                    hook_handles.append(layer.ple.value_proj.register_forward_hook(
                        lambda _m, _i, output: captured.__setitem__(
                            "ple_value_projection", output.detach())))
                    hook_handles.append(layer.ple.norm_key.register_forward_hook(
                        lambda _m, _i, output: captured.__setitem__(
                            "ple_key_normed", output.detach())))
                    hook_handles.append(layer.ple.norm_query.register_forward_hook(
                        lambda _m, _i, output: captured.__setitem__(
                            "ple_query_normed", output.detach())))
                    hook_handles.append(layer.ple.norm_conv.register_forward_hook(
                        lambda _m, _i, output: captured.__setitem__(
                            "ple_gated_value_normed", output.detach())))
                    hook_handles.append(layer.ple.conv1d.register_forward_hook(
                        lambda _m, _i, output: captured.__setitem__(
                            "ple_conv_raw", output.detach())))
                hook_handles.append(
                    layer.attn_hyper_connection.hc_norm.register_forward_hook(
                        lambda _m, _i, output: captured.__setitem__(
                            "core_pre_norm", output.detach())))
                hook_handles.append(layer.attn_hyper_connection.register_forward_hook(
                    lambda _m, _i, output: captured.__setitem__(
                        "attn_hyper", output)))
                hook_handles.append(
                    layer.attn_hyper_connection.register_forward_pre_hook(
                        lambda _m, inputs: captured.__setitem__(
                            "layer_input", inputs[0].detach())))
                core_module = (
                    layer.linear_attn
                    if layer.layer_type == "linear_attention"
                    else layer.self_attn
                )
                hook_handles.append(core_module.register_forward_hook(
                    lambda _m, _i, output: captured.__setitem__(
                        "core_output", first_output(output).detach())))
                hook_handles.append(layer.mlp_hyper_connection.hc_norm.register_forward_hook(
                    lambda _m, _i, output: captured.__setitem__(
                        "mlp_pre_norm", output.detach()) or
                    captured.__setitem__("router_chain_rmsnorm", output.detach())))
                hook_handles.append(
                    layer.mlp_hyper_connection.register_forward_hook(
                        lambda _m, _i, output: captured.__setitem__(
                            "mlp_hyper", output) or
                        captured.__setitem__("router_chain_gr",
                                             output[0].detach())))
                hook_handles.append(layer.mlp.register_forward_hook(
                    lambda _m, _i, output: captured.__setitem__(
                        "mlp_output", output.detach())))
                hook_handles.append(layer.mlp.shared_expert.register_forward_hook(
                    lambda _m, _i, output: captured.__setitem__(
                        "shared_raw", output.detach())))
                hook_handles.append(layer.mlp.shared_expert_gate.register_forward_hook(
                    lambda _m, _i, output: captured.__setitem__(
                        "shared_gate", output.detach())))
                hook_handles.append(layer.mlp_hyper_connection.register_forward_pre_hook(
                    lambda _m, inputs: captured.__setitem__(
                        "router_chain_input", inputs[0].detach())))
            hook_handles.append(layer.mlp.experts.register_forward_hook(
                lambda _m, _i, output: captured.__setitem__(
                    "routed_output", output.detach())))
            qsa_hook = None
            if layer.layer_type != "linear_attention":
                qsa_hook = layer.self_attn.indexer.register_forward_hook(
                    lambda _m, _i, output: captured.__setitem__("qsa", output))
            hidden = layer(hidden, position_embeddings=position_embeddings,
                           attention_mask=full_mask, conv_mask=conv_mask,
                           past_key_values=None, ple_input_ids=input_ids)
            for handle in hook_handles:
                handle.remove()
            if qsa_hook is not None:
                qsa_hook.remove()
            router = captured["router"]
            router_last = tuple(
                value[:, -1] if value.ndim >= 3 else value[-1:]
                for value in router
            )
            logits = router_last[0].reshape(-1).float()
            order = torch.argsort(torch.softmax(logits, 0), descending=True,
                                  stable=True)
            rank = order[:15].cpu().tolist()
            probs = torch.softmax(logits, 0)
            route = {
                "layer": layer_index,
                "hidden_input": values(captured["moe_input"]),
                "router_logits": values(logits),
                "effective_bits": bf16_bits(logits),
                "rank10": {"expert": rank[9], "score": float(logits[rank[9]])},
                "rank11": {"expert": rank[10], "score": float(logits[rank[10]])},
                "margin_rank10_rank11": float(logits[rank[9]] - logits[rank[10]]),
                "top15_rank": [{"rank": i + 1, "expert": e,
                                "value": float(logits[e])}
                               for i, e in enumerate(rank)],
                "top20_rank": [{"rank": i + 1, "expert": e,
                                "value": float(logits[e])}
                               for i, e in enumerate(order[:20].cpu().tolist())],
                "experts": router_last[2].reshape(-1).cpu().tolist(),
                "weights": router_last[1].reshape(-1).float().cpu().tolist(),
                "routed_output": values(captured["routed_output"][-1:]),
            }
            if layer_index == 9:
                native_router_item = next(
                    (x for x in native.get("router_logits", [])
                     if x.get("layer") == layer_index), {})
                native_top = {
                    item.get("id") for item in native_router_item.get("top", [])
                }
                ref_top = set(order[:20].cpu().tolist())
                used_rows = sorted(
                    native_top | ref_top | set(route["experts"])
                )
                gate_name = f"model.language_model.layers.{layer_index}.mlp.gate.weight"
                raw_rows = reader.raw_rows(gate_name, used_rows)
                route["router_chain"] = {
                    "input_to_mlp_hyper_connection":
                        values(captured["router_chain_input"]),
                    "rmsnorm_output": values(captured["router_chain_rmsnorm"]),
                    "gr_output_to_router": values(captured["router_chain_gr"]),
                }
                route["router_chain_stats"] = {
                    name: stats(tensor)
                    for name, tensor in (
                        ("input_to_mlp_hyper_connection",
                         captured["router_chain_input"]),
                        ("rmsnorm_output", captured["router_chain_rmsnorm"]),
                        ("gr_output_to_router", captured["router_chain_gr"]),
                        ("router_matvec_pre_cast",
                         layer.mlp.gate.last_pre_cast),
                        ("router_effective_bf16",
                         layer.mlp.gate.last_effective),
                    )
                }
                route["router_weight_rows"] = [
                    {
                        "expert": expert,
                        "bf16_bytes_sha256": hashlib.sha256(
                            raw_rows[pos]).hexdigest(),
                        "values": values(layer.mlp.gate.weight[expert]),
                        "stats": stats(layer.mlp.gate.weight[expert]),
                    }
                    for pos, expert in enumerate(used_rows)
                ]
                route["router_matvec_pre_cast"] = values(
                    layer.mlp.gate.last_pre_cast)
                route["router_effective_bf16"] = route["router_logits"]
                input_row = captured["moe_input"].reshape(
                    -1, config.hidden_size)[0].float()
                route["router_matvec_checks"] = []
                for expert in used_rows:
                    weight_row = layer.mlp.gate.weight[expert].float()
                    dot = torch.dot(input_row, weight_row)
                    expected = layer.mlp.gate.last_pre_cast[-1, expert]
                    route["router_matvec_checks"].append({
                        "expert": expert,
                        "dot": float(dot),
                        "router_pre_cast": float(expected),
                        "delta": float(dot - expected),
                        "close": math.isclose(float(dot), float(expected),
                                              rel_tol=1e-6, abs_tol=1e-6),
                    })
                route["rounding_diagnostics"] = {
                    "fp32_reduction": values(
                        layer.mlp.gate.last_fp32_reduction),
                    "cuda_matmul_pre_cast": route["router_matvec_pre_cast"],
                    "fp32_cast_effective": route["router_effective_bf16"],
                    "fp32_cast_effective_bits": route["effective_bits"],
                    "bf16_matmul_effective": values(
                        layer.mlp.gate.last_bf16_matmul),
                    "bf16_matmul_effective_bits": bf16_bits(
                        layer.mlp.gate.last_bf16_matmul),
                    "cuda_matmul_effective": route["router_effective_bf16"],
                    "cuda_matmul_effective_bits": route["effective_bits"],
                    "fp32_reduction_vs_cuda_matmul": vector_metrics(
                        values(layer.mlp.gate.last_fp32_reduction),
                        route["router_matvec_pre_cast"]),
                    "bf16_matmul_vs_cuda_matmul": vector_metrics(
                        values(layer.mlp.gate.last_bf16_matmul),
                        route["router_effective_bf16"]),
                    "bf16_matmul_vs_fp32_cast": vector_metrics(
                        values(layer.mlp.gate.last_bf16_matmul),
                        route["router_effective_bf16"]),
                }
                attn_hyper = captured["attn_hyper"]
                mlp_hyper = captured["mlp_hyper"]
                core_residual = (
                        attn_hyper[1]
                        + (
                            captured["core_output"].unsqueeze(-2)
                            * attn_hyper[2].unsqueeze(-1)
                        ).flatten(-2)
                )
                final_mlp = (
                        mlp_hyper[1]
                        + (
                            captured["mlp_output"].unsqueeze(-2)
                            * mlp_hyper[2].unsqueeze(-1)
                        ).flatten(-2)
                )
                shared = (
                        torch.sigmoid(captured["shared_gate"])
                        * captured["shared_raw"]
                )
                boundary_sets[layer_index] = [
                        boundary_entry("layer_input", captured["layer_input"]),
                ]
                boundary_sets[layer_index].extend([
                        boundary_entry("core_pre_norm", captured["core_pre_norm"]),
                        boundary_entry("gr_core_read", attn_hyper[0]),
                        boundary_entry("gdn_qsa_input", attn_hyper[0]),
                        boundary_entry("gdn_qsa_output",
                                       captured["core_output"]),
                        boundary_entry("core_residual_gr_write", core_residual),
                        boundary_entry("mlp_pre_norm", captured["mlp_pre_norm"]),
                        boundary_entry("mlp_gr_read", mlp_hyper[0]),
                        boundary_entry("router_input", captured["moe_input"]),
                        boundary_entry("router_logits_pre_cast",
                                       layer.mlp.gate.last_pre_cast),
                        boundary_entry("router_logits_effective",
                                       layer.mlp.gate.last_effective),
                        boundary_entry("routed_output",
                                       captured["routed_output"]),
                        boundary_entry("shared_expert", shared),
                        boundary_entry("final_mlp_gr_write", final_mlp),
                        boundary_entry("layer_output", hidden),
                ])
                boundary_sets[layer_index].append({
                        "name": "selected_experts",
                        "width": len(route["experts"]),
                        "elements": len(route["experts"]),
                        "ids": route["experts"],
                })
            if layer_index == 1:
                attn_hyper = captured["attn_hyper"]
                mlp_hyper = captured["mlp_hyper"]
                core_residual = (
                    attn_hyper[1]
                    + (
                        captured["core_output"].unsqueeze(-2)
                        * attn_hyper[2].unsqueeze(-1)
                    ).flatten(-2)
                )
                final_mlp = (
                    mlp_hyper[1]
                    + (
                        captured["mlp_output"].unsqueeze(-2)
                        * mlp_hyper[2].unsqueeze(-1)
                    ).flatten(-2)
                )
                shared = (
                    torch.sigmoid(captured["shared_gate"])
                    * captured["shared_raw"]
                )
                boundary_sets[layer_index] = [
                    boundary_entry("layer_input", captured["layer_input"]),
                ]
                if "ple_contribution" in captured:
                    key_normed = captured["ple_key_normed"].unflatten(
                        -1, (config.hc_count, config.hidden_size))
                    query_normed = captured["ple_query_normed"].unflatten(
                        -1, (config.hc_count, config.hidden_size))
                    value = captured["ple_value_projection"]
                    gate = (key_normed * query_normed).sum(
                        dim=-1, keepdim=True) / math.sqrt(config.hidden_size)
                    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
                    gated = torch.sigmoid(gate) * value.unsqueeze(-2)
                    conv_output = torch.nn.functional.silu(
                        captured["ple_conv_raw"]).transpose(1, 2)
                    boundary_sets[layer_index].extend([
                        boundary_entry("ple_embedding",
                                       captured["ple_embedding"]),
                        boundary_entry("ple_key_projection",
                                       captured["ple_key_projection"]),
                        boundary_entry("ple_value_projection",
                                       captured["ple_value_projection"]),
                        boundary_entry("ple_key_normed",
                                       captured["ple_key_normed"]),
                        boundary_entry("ple_query_normed",
                                       captured["ple_query_normed"]),
                        boundary_entry("ple_gated_value",
                                       gated.flatten(-2)),
                        boundary_entry("ple_gated_value_normed",
                                       captured["ple_gated_value_normed"]),
                        boundary_entry("ple_conv_output", conv_output),
                    ])
                    boundary_sets[layer_index].append(
                        boundary_entry("ple_contribution",
                                       captured["ple_contribution"])
                    )
                boundary_sets[layer_index].extend([
                    boundary_entry("core_pre_norm", captured["core_pre_norm"]),
                    boundary_entry("gr_core_read", attn_hyper[0]),
                    boundary_entry("gdn_qsa_input", attn_hyper[0]),
                    boundary_entry("gdn_qsa_output",
                                   captured["core_output"]),
                    boundary_entry("core_residual_gr_write", core_residual),
                    boundary_entry("mlp_pre_norm", captured["mlp_pre_norm"]),
                    boundary_entry("mlp_gr_read", mlp_hyper[0]),
                    boundary_entry("router_input", captured["moe_input"]),
                    boundary_entry("router_logits_pre_cast",
                                   layer.mlp.gate.last_pre_cast),
                    boundary_entry("router_logits_effective",
                                   layer.mlp.gate.last_effective),
                    boundary_entry("routed_output",
                                   captured["routed_output"]),
                    boundary_entry("shared_expert", shared),
                    boundary_entry("final_mlp_gr_write", final_mlp),
                    boundary_entry("layer_output", hidden),
                    {
                        "name": "selected_experts",
                        "width": len(route["experts"]),
                        "elements": len(route["experts"]),
                        "ids": route["experts"],
                    },
                ])
            routing.append(route)
            if layer_index == 2:
                layer2_moe_trace = {
                    "router_input": route["hidden_input"],
                    "router_logits_pre_cast": values(
                        layer.mlp.gate.last_pre_cast),
                    "router_logits_effective": route["router_logits"],
                    "top15_rank": [
                        {"rank": i + 1, "expert": e,
                         "value": float(layer.mlp.gate.last_probs_effective[-1, e])}
                        for i, e in enumerate(rank)
                    ],
                    "margin_rank10_rank11": float(
                        layer.mlp.gate.last_probs_effective[-1, rank[9]] -
                        layer.mlp.gate.last_probs_effective[-1, rank[10]]),
                    "selected_experts": route["experts"],
                    "selected_weights_pre_cast": values(
                        (lambda selected: selected / selected.sum())(
                            layer.mlp.gate.last_probs_pre[0].gather(
                                0, layer.mlp.gate.last_selected[0]))),
                    "selected_weights_effective": route["weights"],
                    "routed_output": route["routed_output"],
                    "router_dtype": "bf16",
                }
            native_route = next((x for x in native.get("routing", [])
                                 if x.get("layer") == layer_index), None)
            native_logits = native_router(native, layer_index)
            native_router_item = next(
                (x for x in native.get("router_logits", [])
                 if x.get("layer") == layer_index), None)
            native_experts = native_route.get("experts", []) if native_route else None
            ref_experts = route["experts"]
            if (layer_index >= args.start_layer and native_experts is not None
                    and set(native_experts) != set(ref_experts)
                    and first_divergence is None):
                first_divergence = layer_index
            native_weight_map = dict(zip(
                native_experts or [],
                native_route.get("weights", []) if native_route else []
            ))
            ref_weight_map = dict(zip(ref_experts, route["weights"]))
            common_experts = sorted(set(native_weight_map) & set(ref_weight_map))
            weight_check = vector_metrics(
                [native_weight_map[e] for e in common_experts],
                [ref_weight_map[e] for e in common_experts],
            )
            if set(native_weight_map) != set(ref_weight_map):
                weight_check = {"status": "fail", "reason": "expert-set"}
            route["native_reference"] = {
                "selected_set_exact": (set(native_experts) == set(ref_experts)
                                       if native_experts is not None else None),
                "weights_by_expert": weight_check
                    if native_experts is not None else {"status": "missing"},
                "router_logits": vector_metrics(native_logits,
                                                route["router_logits"]),
            }
            if native_route and native_route.get("rank10") and native_route.get("rank11"):
                route["native_reference"].update({
                    "native_rank10": native_route["rank10"],
                    "native_rank11": native_route["rank11"],
                    "native_margin_rank10_rank11": native_route.get(
                        "margin_rank10_rank11"),
                })
            elif native_router_item:
                route["native_reference"].update({
                    "native_rank10": native_router_item.get("rank10"),
                    "native_rank11": native_router_item.get("rank11"),
                    "native_margin_rank10_rank11": native_router_item.get(
                        "margin_rank10_rank11"),
                })
            if "qsa" in captured:
                qsa_selection.append({
                    "layer": layer_index,
                    "selected": torch.where(captured["qsa"][0, -1, 0].bool())[0]
                    .cpu().tolist(),
                })
            stages.append({
                "stage": "layer", "layer": layer_index, "width": hidden.shape[-1],
                "stats": stats(hidden),
                "values": values(hidden),
            })
            del layer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    with torch.device("meta"):
        mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
    prefix = "model.language_model.hyper_connection_mixer."
    for name, _ in list(mixer.named_parameters()):
        replace_parameter(mixer, name, reader.dense(prefix + name, device))
    mixer = mixer.to(device)
    final_norm = mixer(hidden)
    stages.append({"stage": "final_norm", "layer": 0, "width": final_norm.shape[-1],
                   "stats": stats(final_norm)})
    lm_head = reader.dense("lm_head.weight", device)
    logits = torch.matmul(final_norm.float(), lm_head.float().transpose(0, 1)).reshape(-1)
    stages.append({"stage": "logits", "layer": 0, "width": logits.numel(),
                   "stats": stats(logits), "top": top_k(logits)})
    report = {
        "format": "q38-m6-quant-matched-reference-v3",
        "reference": "independent GGUF reader/dequantizer + Qwen4Exp math",
        "gguf": str(args.gguf), "tokens": tokens, "q38_forward_called": False,
        "comparison_start_layer": args.start_layer,
        "stages": stages, "routing": routing, "qsa_selection": qsa_selection,
        "layer1_boundaries": boundary_sets.get(1, []),
        "layer9_boundaries": boundary_sets.get(9, []),
        "layer9_ple_contribution": {
            "status": "not_applicable",
            "reason": "layer 9 has no PLE",
        },
        "layer2_moe_trace": layer2_moe_trace,
        "first_selected_set_divergence_layer": first_divergence,
        "status": "pass" if all(s["stats"]["nan_count"] == 0 and
                                s["stats"]["inf_count"] == 0 for s in stages)
        else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    reader.close()
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
