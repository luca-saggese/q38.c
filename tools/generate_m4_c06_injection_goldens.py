#!/usr/bin/env python3
"""Generate independent, checkpoint-backed PLE injection vectors."""

import argparse
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open


HIDDEN = 2560
STREAMS = 4
HEADS = 16
ROW_WIDTH = 160
CHANNELS = HIDDEN * STREAMS
KERNEL = 4
DILATION = 3
EOS = 248044


class Reader:
    def __init__(self, root: Path):
        self.root = root
        self.index = json.loads(
            (root / "model.safetensors.index.json").read_text()
        )["weight_map"]

    def get(self, name):
        with safe_open(str(self.root / self.index[name]), framework="pt") as h:
            return h.get_tensor(name).float().contiguous()

    def rows(self, name, rows):
        with safe_open(str(self.root / self.index[name]), framework="pt") as h:
            return h.get_slice(name)[rows].float().contiguous()


def row_ids(tokens, multipliers, offsets, sizes):
    history = [EOS, EOS]
    result = []
    for token in tokens:
        context = [token, history[-1], history[-2]]
        ids = []
        for n in (2, 3):
            mixed = context[0] * multipliers[0]
            for j in range(1, n):
                mixed ^= context[j] * multipliers[j]
            base = 0 if n == 2 else 8
            ids.extend(
                int(mixed % sizes[base + g] + offsets[base + g])
                for g in range(8)
            )
        result.append(ids)
        history.append(token)
    return result


def grouped_norm(x, weight):
    x = x.view(x.shape[0], STREAMS, HIDDEN)
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    return (x * weight.view(1, STREAMS, HIDDEN)).flatten(1)


def run(root: Path, tokens):
    r = Reader(root)
    offsets = r.get(
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets"
    ).long().tolist()
    sizes = r.get(
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"
    ).long().tolist()
    multipliers = r.get(
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"
    ).long().tolist()
    ids = row_ids(tokens, multipliers, offsets, sizes)

    rows = []
    for token_ids in ids:
        row_group = []
        for row in token_ids:
            shard = row // 2_500_012
            local = row % 2_500_012
            name = (
                "model.language_model.layers.1.ple.ple_embedding."
                f"ngram_embedding.shard_{shard}.weight"
            )
            row_group.append(r.rows(name, [local])[0])
        rows.append(torch.cat(row_group))
    embedding = torch.stack(rows)

    hidden = r.rows("model.language_model.embed_tokens.weight", tokens)
    hidden = hidden[:, None, :].expand(-1, STREAMS, -1).flatten(1).contiguous()
    key_proj = r.get("model.language_model.layers.1.ple.key_proj.weight")
    value_proj = r.get("model.language_model.layers.1.ple.value_proj.weight")
    norm_key = r.get("model.language_model.layers.1.ple.norm_key.weight")
    norm_query = r.get("model.language_model.layers.1.ple.norm_query.weight")
    norm_conv = r.get("model.language_model.layers.1.ple.norm_conv.weight")
    conv_raw = r.get("model.language_model.layers.1.ple.conv1d.weight")
    conv = conv_raw[:, 0, :].T.contiguous()

    key = embedding @ key_proj.T
    value = embedding @ value_proj.T
    query = grouped_norm(hidden, norm_query)
    key = grouped_norm(key, norm_key)
    scores = (key * query).view(-1, STREAMS, HIDDEN).sum(-1) / HIDDEN**0.5
    gate = torch.sigmoid(torch.sign(scores) * scores.abs().clamp_min(1e-6).sqrt())
    gated = value[:, None, :] * gate[:, :, None]
    normalized = grouped_norm(gated.flatten(1), norm_conv).view(-1, CHANNELS)
    history = torch.zeros((DILATION * (KERNEL - 1), CHANNELS))
    padded = torch.cat((history, normalized))
    conv_out = torch.zeros_like(normalized)
    for t in range(len(tokens)):
        for k in range(KERNEL):
            src = DILATION * (KERNEL - 1) + t - (KERNEL - 1 - k) * DILATION
            conv_out[t] += conv[k] * padded[src]
    conv_out = torch.nn.functional.silu(conv_out)
    contribution = gated.flatten(1) + conv_out
    after = hidden + contribution
    return ids, hidden, embedding, key_proj, value_proj, norm_key, norm_query, norm_conv, conv, contribution, after


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    args = p.parse_args()
    tokens = [EOS, 9707, 11, 576]
    values = run(args.model_dir, tokens)
    ids, hidden, embedding, key_proj, value_proj, nk, nq, nc, conv, contribution, after = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "format": "q38-m4-c06-ple-injection-v1", "status": "pass",
        "tokens": tokens, "row_ids": ids, "layer": 1,
        "input": "checkpoint embed_tokens repeated across four streams",
        "reference": "official Qwen4Exp PLE equations",
        "vectors": {"hidden_before_ple": True, "ple_contribution": True,
                    "hidden_after_ple": True},
    }, indent=2) + "\n")
    arrays = [hidden, embedding, key_proj, value_proj, nk, nq, nc, conv,
              contribution, after]
    with args.fixture.open("wb") as f:
        f.write(struct.pack("<4sIIII", b"P38F", len(tokens), CHANNELS,
                            HEADS * ROW_WIDTH, len(arrays)))
        for a in arrays:
            f.write(a.numpy().astype("float32").tobytes())
        for row_group in ids:
            f.write(struct.pack("<" + "I" * len(row_group), *row_group))


if __name__ == "__main__":
    main()
