#!/usr/bin/env python3
"""Run the official Qwen4-Exp text graph one layer at a time.

The checkpoint is too large for a whole-model load on the Spark host.  This
runner therefore materializes only the current Transformers layer, while
keeping the reference equations in the official implementation.  It records
complete-vector statistics and exact routing decisions; it never imports or
calls q38.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import Qwen4ExpTextConfig

_qwen4 = __import__(
    "transformers.models.qwen4_exp.modeling_qwen4_exp",
    fromlist=[
        "Qwen4ExpTextDecoderLayer",
        "Qwen4ExpTextGatedResidual",
        "Qwen4ExpTextRotaryEmbedding",
    ],
)
Qwen4ExpTextDecoderLayer = _qwen4.Qwen4ExpTextDecoderLayer
Qwen4ExpTextGatedResidual = _qwen4.Qwen4ExpTextGatedResidual
Qwen4ExpTextRotaryEmbedding = _qwen4.Qwen4ExpTextRotaryEmbedding


FIXED = (0, 1, 2, 3)
DEVICE = torch.device("cuda")


def stats(values: torch.Tensor) -> dict:
    flat = values.detach().float().reshape(-1).cpu()
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    nan_count = int(torch.isnan(flat).sum())
    inf_count = int(torch.isinf(flat).sum())
    finite_count = int(finite.sum())
    raw = flat.numpy().tobytes()
    digest = hashlib.sha256(raw).hexdigest()
    if finite_values.numel():
        minimum = float(finite_values.min())
        maximum = float(finite_values.max())
        mean = float(finite_values.mean())
        rms = float(torch.sqrt(torch.mean(finite_values * finite_values)))
        max_abs = float(torch.max(torch.abs(finite_values)))
        finite_indices = torch.where(finite)[0]
        min_index = int(finite_indices[torch.argmin(finite_values)])
        max_index = int(finite_indices[torch.argmax(finite_values)])
        max_abs_index = int(finite_indices[torch.argmax(torch.abs(finite_values))])
    else:
        minimum = maximum = mean = rms = max_abs = None
        min_index = max_index = max_abs_index = None
    fixed = [
        {"index": i, "value": float(flat[i]) if i < flat.numel() else None}
        for i in FIXED
    ]
    return {
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "rms": rms,
        "max_abs": max_abs,
        "finite_count": finite_count,
        "min_index": min_index,
        "max_index": max_index,
        "max_abs_index": max_abs_index,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "checksum": digest,
        "fixed": fixed,
    }


class Checkpoint:
    def __init__(self, root: Path):
        self.root = root
        self.weight_map = json.loads(
            (root / "model.safetensors.index.json").read_text()
        )["weight_map"]

    def get(self, name: str, device: torch.device = DEVICE) -> torch.Tensor:
        path = self.root / self.weight_map[name]
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            return stream.get_tensor(name).to(device)

    def slice(self, name: str, start: int, end: int) -> torch.Tensor:
        path = self.root / self.weight_map[name]
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            return stream.get_slice(name)[start:end].to(DEVICE)


class StreamingNGramEmbedding(nn.Module):
    """Embedding lookup that reads only the rows used by this token."""

    def __init__(self, checkpoint: Checkpoint, names: list[str]):
        super().__init__()
        self.checkpoint = checkpoint
        self.register_buffer("weight_stub", torch.empty(0, device=DEVICE))
        self.names = sorted(
            names,
            key=lambda name: int(name.rsplit("shard_", 1)[1].split(".", 1)[0]),
        )
        self.starts: list[int] = []
        total = 0
        for name in self.names:
            path = checkpoint.root / checkpoint.weight_map[name]
            with safe_open(str(path), framework="pt", device="cpu") as stream:
                size = stream.get_slice(name).get_shape()[0]
            self.starts.append(total)
            total += size

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_stub

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1)
        output = torch.empty(
            (flat.numel(), 160), dtype=torch.bfloat16, device=DEVICE
        )
        for index, value in enumerate(flat.tolist()):
            shard = 0
            while shard + 1 < len(self.starts) and self.starts[shard + 1] <= value:
                shard += 1
            local = value - self.starts[shard]
            output[index] = self.checkpoint.slice(
                self.names[shard], local, local + 1
            )[0]
        return output.reshape(*ids.shape, 160)


class PlainExperts(nn.Module):
    """The same sparse expert equation without the optional dispatch layer."""

    def __init__(self, source: nn.Module):
        super().__init__()
        self.gate_up_proj = source.gate_up_proj
        self.down_proj = source.down_proj
        self.act_fn = source.act_fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros_like(hidden_states)
        mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.gate_up_proj.shape[0]
        ).permute(2, 1, 0)
        for expert in torch.where(mask.sum(dim=(-1, -2)) > 0)[0]:
            positions, rows = torch.where(mask[expert])
            current = hidden_states[rows]
            gate, up = torch.nn.functional.linear(
                current, self.gate_up_proj[expert]
            ).chunk(2, dim=-1)
            current = self.act_fn(gate) * up
            current = torch.nn.functional.linear(current, self.down_proj[expert])
            current = current * top_k_weights[rows, positions, None]
            result.index_add_(0, rows, current.to(result.dtype))
        return result


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


def materialize_layer(
    checkpoint: Checkpoint, config: Qwen4ExpTextConfig, layer_index: int
) -> nn.Module:
    with torch.device("meta"):
        layer = Qwen4ExpTextDecoderLayer(config, layer_index)
    prefix = f"model.language_model.layers.{layer_index}."
    for name, _ in list(layer.named_parameters()):
        if "ple_embedding.ngram_embedding.weight" in name:
            continue
        replace_parameter(layer, name, checkpoint.get(prefix + name))
    for name, _ in list(layer.named_buffers()):
        full_name = prefix + name
        if full_name in checkpoint.weight_map:
            replace_buffer(layer, name, checkpoint.get(full_name))
    if layer.ple is not None:
        names = [
            name
            for name in checkpoint.weight_map
            if prefix + "ple.ple_embedding.ngram_embedding.shard_" in name
        ]
        layer.ple.ple_embedding.ngram_embedding = StreamingNGramEmbedding(
            checkpoint, names
        )
    layer.mlp.experts = PlainExperts(layer.mlp.experts)
    return layer


def top_k(values: torch.Tensor, count: int = 10) -> list[dict]:
    flat = values.detach().float().cpu().tolist()
    order = sorted(range(len(flat)), key=lambda i: (-flat[i], i))[:count]
    return [{"id": i, "value": flat[i]} for i in order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    tokens = trace.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise SystemExit("trace must declare its non-empty token sequence")
    if any(not isinstance(token, int) for token in tokens):
        raise SystemExit("trace tokens must be integers")

    raw_config = json.loads((args.model_dir / "config.json").read_text())
    config = Qwen4ExpTextConfig(**raw_config["text_config"])
    checkpoint = Checkpoint(args.model_dir)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    embedding_name = "model.language_model.embed_tokens.weight"
    hidden = checkpoint.slice(embedding_name, tokens[0], tokens[0] + 1)
    hidden = hidden.reshape(1, 1, config.hidden_size).repeat(1, 1, config.hc_count)
    rotary = Qwen4ExpTextRotaryEmbedding(config).to(DEVICE)
    position_ids = torch.zeros((3, 1, len(tokens)), dtype=torch.long, device=DEVICE)
    position_embeddings = rotary(hidden, position_ids)
    full_mask = torch.ones(
        (1, 1, len(tokens), len(tokens)), dtype=torch.bool, device=DEVICE
    )
    conv_mask = torch.ones((1, len(tokens)), dtype=torch.bool, device=DEVICE)
    stages = []
    routing = []
    qsa_selection = []
    layer2_moe_trace = None

    for layer_index in range(config.num_hidden_layers):
        print(f"reference layer {layer_index}", flush=True)
        layer = materialize_layer(checkpoint, config, layer_index).to(DEVICE)
        captured = {}

        def capture(_module, _inputs, output):
            captured["router"] = output

        hook = layer.mlp.gate.register_forward_hook(capture)
        moe_input_hook = layer.mlp.register_forward_pre_hook(
            lambda _module, inputs: captured.__setitem__(
                "moe_input", inputs[0].detach()
            )
        )
        routed_output_hook = layer.mlp.experts.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__(
                "routed_output", output.detach()
            )
        )
        qsa_hook = None
        if layer.layer_type != "linear_attention":
            def capture_qsa(_module, _inputs, output):
                captured["qsa"] = output

            qsa_hook = layer.self_attn.indexer.register_forward_hook(capture_qsa)
        with torch.no_grad():
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=full_mask,
                conv_mask=conv_mask,
                past_key_values=None,
                ple_input_ids=input_ids,
            )
        hook.remove()
        moe_input_hook.remove()
        routed_output_hook.remove()
        if qsa_hook is not None:
            qsa_hook.remove()
        router = captured.get("router")
        if router is None:
            raise RuntimeError(f"router output missing at layer {layer_index}")
        routing.append(
            {
                "layer": layer_index,
                "experts": router[2].reshape(-1, config.num_experts_per_tok)
                .cpu()
                .tolist(),
                "weights": router[1].reshape(-1, config.num_experts_per_tok)
                .float()
                .cpu()
                .tolist(),
                "logits_top": top_k(router[0].reshape(-1)),
            }
        )
        if layer_index == 2:
            moe_input = captured["moe_input"].reshape(-1, config.hidden_size)
            effective_logits = router[0].reshape(-1).float()
            pre_cast_logits = torch.nn.functional.linear(
                moe_input.float(), layer.mlp.gate.weight.float()
            ).reshape(-1)
            effective_probs = torch.softmax(effective_logits, dim=-1)
            selected = router[2].reshape(-1)
            selected_pre_cast = effective_probs.index_select(0, selected)
            selected_pre_cast = selected_pre_cast / selected_pre_cast.sum()
            selected_effective = router[1].reshape(-1).float()
            order = sorted(
                range(config.num_experts),
                key=lambda i: (-float(effective_probs[i]), i),
            )
            layer2_moe_trace = {
                "router_input": moe_input[0].cpu().tolist(),
                "router_logits_pre_cast": pre_cast_logits.cpu().tolist(),
                "router_logits_effective": effective_logits.cpu().tolist(),
                "top15_rank": [
                    {"rank": rank + 1, "expert": expert,
                     "value": float(effective_probs[expert])}
                    for rank, expert in enumerate(order[:15])
                ],
                "margin_rank10_rank11": float(
                    effective_probs[order[9]] - effective_probs[order[10]]
                ),
                "selected_experts": selected.cpu().tolist(),
                "selected_weights_pre_cast": selected_pre_cast.cpu().tolist(),
                "selected_weights_effective": selected_effective.cpu().tolist(),
                "routed_output": captured["routed_output"].reshape(-1)[0:config.hidden_size]
                .cpu()
                .tolist(),
                "router_dtype": str(router[0].dtype).replace("torch.", ""),
            }
        if "qsa" in captured:
            visible = captured["qsa"][0, 0, 0].bool().cpu()
            qsa_selection.append(
                {"layer": layer_index, "selected": torch.where(visible)[0].tolist()}
            )
        stages.append(
            {
                "stage": "layer",
                "layer": layer_index,
                "width": hidden.shape[-1],
                "stats": stats(hidden),
            }
        )
        del layer, router
        torch.cuda.empty_cache()

    with torch.device("meta"):
        mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
    prefix = "model.language_model.hyper_connection_mixer."
    for name, _ in list(mixer.named_parameters()):
        replace_parameter(mixer, name, checkpoint.get(prefix + name))
    mixer = mixer.to(DEVICE)
    with torch.no_grad():
        final_norm = mixer(hidden)
    stages.append(
        {
            "stage": "final_norm",
            "layer": 0,
            "width": final_norm.shape[-1],
            "stats": stats(final_norm),
        }
    )
    lm_head = checkpoint.get("lm_head.weight")
    with torch.no_grad():
        logits = torch.matmul(final_norm.float(), lm_head.float().transpose(0, 1))
    logits = logits.reshape(-1)
    stages.append(
        {
            "stage": "logits",
            "layer": 0,
            "width": logits.numel(),
            "stats": stats(logits),
            "top": top_k(logits),
        }
    )
    report = {
        "format": "q38-m6-transformers-reference-v3",
        "reference": "official Transformers Qwen4Exp implementation",
        "model_dir": str(args.model_dir),
        "tokens": tokens,
        "stages": stages,
        "routing": routing,
        "qsa_selection": qsa_selection,
        "layer2_moe_trace": layer2_moe_trace,
        "status": "pass"
        if all(
            stage["stats"]["nan_count"] == 0 and stage["stats"]["inf_count"] == 0
            for stage in stages
        )
        else "fail",
        "tolerance": {"absolute": 0.02, "relative": 0.002},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
