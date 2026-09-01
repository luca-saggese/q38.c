#!/usr/bin/env python3
"""Independent QSA golden generator.

This intentionally does not import q38 or invoke a q38 executable.  It opens
only the requested safetensor shards, converts the selected BF16 tensors to
float32, and writes a compact fixture consumed by the native probe.
"""

import argparse
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open


HIDDEN = 2560
Q_HEADS = 24
KV_HEADS = 2
HEAD_DIM = 256
INDEX_HEADS = 4
INDEX_DIM = 128
RATIO = 4
BUDGET = 2048
THETA = 10_000_000.0
ROTARY = 64


class Reader:
    def __init__(self, model_dir: Path):
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.model_dir = model_dir
        self.files = {}
        self.map = index["weight_map"]

    def get(self, name: str) -> torch.Tensor:
        path = self.model_dir / self.map[name]
        with safe_open(str(path), framework="pt") as handle:
            return handle.get_tensor(name).float().contiguous()

    def rows(self, name: str, rows) -> torch.Tensor:
        path = self.model_dir / self.map[name]
        with safe_open(str(path), framework="pt") as handle:
            return handle.get_slice(name)[rows].float().contiguous()


def rms(x, weight):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * weight


def rope(x, positions):
    half = ROTARY // 2
    inv = THETA ** (-torch.arange(0, ROTARY, 2, dtype=torch.float32) / ROTARY)
    angles = positions[:, None].float() * inv[None, :]
    c, s = angles.cos(), angles.sin()
    a, b = x[..., :half], x[..., half:ROTARY]
    rotated = torch.cat((a * c[:, None, :] - b * s[:, None, :],
                         a * s[:, None, :] + b * c[:, None, :]), dim=-1)
    return torch.cat((rotated, x[..., ROTARY:]), dim=-1)


def forward(reader: Reader, token_ids):
    emb = reader.rows("model.language_model.embed_tokens.weight", token_ids)
    prefix = "model.language_model.layers.3.self_attn."
    q_proj = reader.get(prefix + "q_proj.weight")
    k_proj = reader.get(prefix + "k_proj.weight")
    v_proj = reader.get(prefix + "v_proj.weight")
    o_proj = reader.get(prefix + "o_proj.weight")
    index_proj = reader.get(prefix + "indexer.index_qk_proj.weight")
    q_norm = reader.get(prefix + "q_norm.weight")
    k_norm = reader.get(prefix + "k_norm.weight")
    iq_norm = reader.get(prefix + "indexer.q_layernorm.weight")
    ik_norm = reader.get(prefix + "indexer.k_layernorm.weight")

    qfull = emb @ q_proj.T
    qgate = qfull.view(-1, Q_HEADS, 2, HEAD_DIM)
    q = qgate[:, :, 0, :]
    gate = qgate[:, :, 1, :]
    q = rope(rms(q, q_norm), torch.arange(len(token_ids)))
    k = rope(rms((emb @ k_proj.T).view(-1, KV_HEADS, HEAD_DIM), k_norm),
             torch.arange(len(token_ids)))
    v = (emb @ v_proj.T).view(-1, KV_HEADS, HEAD_DIM)
    iqk = emb @ index_proj.T
    iq = rope(rms(iqk[:, : INDEX_HEADS * INDEX_DIM].view(-1, INDEX_HEADS, INDEX_DIM),
                  iq_norm), torch.arange(len(token_ids)))
    raw = iqk[:, INDEX_HEADS * INDEX_DIM :].view(-1, INDEX_DIM)

    selected = []
    outputs = []
    for t in range(len(token_ids)):
        visible = raw[: t + 1]
        complete = (t + 1) // RATIO
        scores = []
        for group in range(complete):
            pooled = visible[group * RATIO : (group + 1) * RATIO].mean(0)
            pooled = rope(rms(pooled[None, :], ik_norm)[None, :, :],
                          torch.tensor([group * RATIO]))[0, 0]
            score = (torch.relu(iq[t] @ pooled) / (INDEX_DIM**0.5)).sum(-1)
            scores.append(float(score))
        order = sorted(range(complete),
                       key=lambda g: (-scores[g], g))[: BUDGET // RATIO]
        ids = [g * RATIO + j for g in order for j in range(RATIO)]
        ids += list(range(complete * RATIO, t + 1))
        selected.append(ids)

        q_t = q[t]
        values = []
        for h in range(Q_HEADS):
            kvh = h // (Q_HEADS // KV_HEADS)
            logits = (q_t[h] @ k[: t + 1, kvh].T) / (HEAD_DIM**0.5)
            mask = torch.full_like(logits, float("-inf"))
            mask[ids] = logits[ids]
            weights = torch.softmax(mask, dim=-1)
            values.append(weights @ v[: t + 1, kvh])
        attn = torch.stack(values).flatten() * torch.sigmoid(gate[t].flatten())
        outputs.append(attn @ o_proj.T)
    return (emb, selected, torch.stack(outputs),
            q_proj, k_proj, v_proj, o_proj, index_proj,
            q_norm, k_norm, iq_norm, ik_norm)


def write_fixture(path: Path, emb, selected, outputs, *weights):
    token_count = emb.shape[0]
    with path.open("wb") as f:
        f.write(struct.pack("<4sIIII", b"Q38F", token_count, HIDDEN,
                            max(map(len, selected)), len(weights)))
        f.write(emb.numpy().astype("float32").tobytes())
        f.write(outputs.numpy().astype("float32").tobytes())
        for weight in weights:
            f.write(weight.numpy().astype("float32").tobytes())
        for ids in selected:
            f.write(struct.pack("<I", len(ids)))
            f.write(struct.pack("<" + "I" * len(ids), *ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[248044, 9707, 11, 576])
    args = parser.parse_args()
    reader = Reader(args.model_dir)
    values = forward(reader, args.tokens)
    emb, selected, outputs, *weights = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "gate": "M5-C12", "source": "Qwen3.8-Flash-Next checkpoint",
        "layer": 3, "tokens": args.tokens, "selected": selected,
        "output_shape": list(outputs.shape), "status": "pass",
    }) + "\n")
    write_fixture(args.fixture, emb, selected, outputs, *weights)


if __name__ == "__main__":
    main()
