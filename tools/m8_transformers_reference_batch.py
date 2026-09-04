#!/usr/bin/env python3
"""Run the Transformers Qwen4-Exp reference over a deterministic corpus.

Layers are materialized once and then reused for every sequence before the
next layer is loaded.  Sequences are intentionally processed one at a time:
the Qwen4-Exp attention and convolution masks have sequence-specific shapes,
and this preserves the single-sequence reference semantics without requiring
padding or a batched cache implementation.

The output is JSONL with one compact record per input sequence.  Each record
contains the same observable reference data as the official runner:
hidden-stage statistics, top logits and target logit, routing decisions, QSA
selections, and finite status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .m6_transformers_reference import (
        Checkpoint,
        DEVICE,
        Qwen4ExpTextConfig,
        Qwen4ExpTextGatedResidual,
        Qwen4ExpTextRotaryEmbedding,
        materialize_layer,
        replace_parameter,
        stats,
        top_k,
    )
except ImportError:
    from m6_transformers_reference import (
        Checkpoint,
        DEVICE,
        Qwen4ExpTextConfig,
        Qwen4ExpTextGatedResidual,
        Qwen4ExpTextRotaryEmbedding,
        materialize_layer,
        replace_parameter,
        stats,
        top_k,
    )


def read_corpus(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise SystemExit(f"corpus line {line_number} must be an object")
            tokens = item.get("token_ids")
            if not isinstance(tokens, list) or not tokens:
                raise SystemExit(
                    f"corpus line {line_number} must contain non-empty token_ids"
                )
            if any(not isinstance(token, int) for token in tokens):
                raise SystemExit(
                    f"corpus line {line_number} token_ids must be integers"
                )
            records.append(
                {
                    "id": item.get("id", f"sequence-{len(records):04d}"),
                    "tokens": tokens,
                }
            )
    if not records:
        raise SystemExit("corpus must contain at least one sequence")
    return records


def make_state(checkpoint: Checkpoint, config: Qwen4ExpTextConfig, tokens: list[int]):
    input_ids = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    hidden = checkpoint.slice(
        "model.language_model.embed_tokens.weight", tokens[0], tokens[0] + 1
    )
    hidden = hidden.reshape(1, 1, config.hidden_size).repeat(1, 1, config.hc_count)
    position_ids = torch.zeros((3, 1, len(tokens)), dtype=torch.long, device=DEVICE)
    full_mask = torch.ones(
        (1, 1, len(tokens), len(tokens)), dtype=torch.bool, device=DEVICE
    )
    conv_mask = torch.ones((1, len(tokens)), dtype=torch.bool, device=DEVICE)
    return {
        "id": None,
        "tokens": tokens,
        "input_ids": input_ids,
        "hidden": hidden,
        "position_ids": position_ids,
        "full_mask": full_mask,
        "conv_mask": conv_mask,
        "stages": [],
        "routing": [],
        "qsa_selection": [],
    }


def run(args: argparse.Namespace) -> list[dict]:
    corpus = read_corpus(args.corpus)
    raw_config = json.loads((args.model_dir / "config.json").read_text())
    config = Qwen4ExpTextConfig(**raw_config["text_config"])
    checkpoint = Checkpoint(args.model_dir)
    rotary = Qwen4ExpTextRotaryEmbedding(config).to(DEVICE)
    states = [
        make_state(checkpoint, config, item["tokens"]) for item in corpus
    ]
    for state, item in zip(states, corpus):
        state["id"] = item["id"]
        with torch.no_grad():
            state["position_embeddings"] = rotary(
                state["hidden"], state["position_ids"]
            )

    for layer_index in range(config.num_hidden_layers):
        print(f"reference layer {layer_index} ({len(states)} sequences)", flush=True)
        layer = materialize_layer(checkpoint, config, layer_index).to(DEVICE)
        layer.eval()
        captured = {}

        def capture_router(_module, _inputs, output):
            captured["router"] = output

        hook = layer.mlp.gate.register_forward_hook(capture_router)
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
            qsa_hook = layer.self_attn.indexer.register_forward_hook(
                lambda _module, _inputs, output: captured.__setitem__(
                    "qsa", output
                )
            )

        for state in states:
            captured.clear()
            with torch.no_grad():
                state["hidden"] = layer(
                    state["hidden"],
                    position_embeddings=state["position_embeddings"],
                    attention_mask=state["full_mask"],
                    conv_mask=state["conv_mask"],
                    past_key_values=None,
                    ple_input_ids=state["input_ids"],
                )
            router = captured.get("router")
            if router is None:
                raise RuntimeError(f"router output missing at layer {layer_index}")
            state["routing"].append(
                {
                    "layer": layer_index,
                    "experts": router[2]
                    .reshape(-1, config.num_experts_per_tok)
                    .cpu()
                    .tolist(),
                    "weights": router[1]
                    .reshape(-1, config.num_experts_per_tok)
                    .float()
                    .cpu()
                    .tolist(),
                    "logits_top": top_k(router[0].reshape(-1)),
                }
            )
            if "qsa" in captured:
                visible = captured["qsa"][0, 0, 0].bool().cpu()
                state["qsa_selection"].append(
                    {
                        "layer": layer_index,
                        "selected": torch.where(visible)[0].tolist(),
                    }
                )
            state["stages"].append(
                {
                    "stage": "layer",
                    "layer": layer_index,
                    "width": state["hidden"].shape[-1],
                    "stats": stats(state["hidden"]),
                }
            )
        hook.remove()
        moe_input_hook.remove()
        routed_output_hook.remove()
        if qsa_hook is not None:
            qsa_hook.remove()
        del layer
        torch.cuda.empty_cache()

    with torch.device("meta"):
        mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
    prefix = "model.language_model.hyper_connection_mixer."
    for name, _ in list(mixer.named_parameters()):
        replace_parameter(mixer, name, checkpoint.get(prefix + name))
    mixer = mixer.to(DEVICE)
    lm_head = checkpoint.get("lm_head.weight")

    reports = []
    for state in states:
        with torch.no_grad():
            final_norm = mixer(state["hidden"])
            logits = torch.matmul(
                final_norm.float(), lm_head.float().transpose(0, 1)
            ).reshape(-1)
        state["stages"].append(
            {
                "stage": "final_norm",
                "layer": 0,
                "width": final_norm.shape[-1],
                "stats": stats(final_norm),
            }
        )
        state["stages"].append(
            {
                "stage": "logits",
                "layer": 0,
                "width": logits.numel(),
                "stats": stats(logits),
                "top": top_k(logits, 20),
                "target_logit": float(logits[state["tokens"][-1]]),
            }
        )
        finite = all(
            stage["stats"]["nan_count"] == 0
            and stage["stats"]["inf_count"] == 0
            for stage in state["stages"]
        )
        reports.append(
            {
                "format": "q38-m8-transformers-reference-batch-v1",
                "reference": "official Transformers Qwen4Exp implementation",
                "sequence_id": state["id"],
                "tokens": state["tokens"],
                "stages": state["stages"],
                "routing": state["routing"],
                "qsa_selection": state["qsa_selection"],
                "status": "pass" if finite else "fail",
                "finite": finite,
                "tolerance": {"absolute": 0.02, "relative": 0.002},
            }
        )
    return reports


def check_reference(report: dict, reference_path: Path) -> None:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    # The official runner includes an optional full layer-7 value trace;
    # compact corpus records intentionally retain only its stats/checksum.
    compact_stages = [
        {key: value for key, value in stage.items() if key != "values"}
        for stage in report["stages"]
    ]
    reference_stages = [
        {key: value for key, value in stage.items() if key != "values"}
        for stage in reference["stages"]
    ]
    checks = {
        "tokens": report["tokens"] == reference["tokens"],
        "stages": compact_stages == reference_stages,
        "routing": report["routing"] == reference["routing"],
        "qsa_selection": report["qsa_selection"] == reference["qsa_selection"],
        "status": report["status"] == reference["status"],
    }
    for field, matches in checks.items():
        if not matches:
            raise SystemExit(f"self-check failed for {field}")
    print("self-check: first corpus record matches official reference", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-check", type=Path)
    args = parser.parse_args()
    reports = run(args)
    if args.self_check:
        check_reference(reports[0], args.self_check)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for report in reports:
            stream.write(json.dumps(report) + "\n")
    if any(report["status"] != "pass" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
