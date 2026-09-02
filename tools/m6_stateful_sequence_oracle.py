#!/usr/bin/env python3
"""Stateful official Qwen4Exp oracle with explicit CUDA evidence.

The graph and cache are supplied by Transformers.  GGUF parsing/dequantization
is CPU work; persistent decoder weights, activations, the official
DynamicCache, and the LM head live on the requested device.  Decoder layers
are materialized once, while routed expert matrices use a bounded CUDA LRU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.m6_stateful_gguf_oracle import (  # noqa: E402
    cache_evidence,
    compare_stats,
    load_reference,
    materialize_meta_model,
    stats,
    top_k,
)


class Timing:
    def __init__(self) -> None:
        self.gguf_read_ms = 0.0
        self.dequant_ms = 0.0
        self.h2d_ms = 0.0

    def snapshot(self) -> dict[str, float]:
        return {
            "gguf_read_ms": self.gguf_read_ms,
            "dequant_ms": self.dequant_ms,
            "h2d_ms": self.h2d_ms,
        }

    def delta(self, before: dict[str, float]) -> dict[str, float]:
        return {
            key: round(getattr(self, key) - value, 3)
            for key, value in before.items()
        }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def instrument_cache(cache: object, device: torch.device) -> dict[str, float | int]:
    """Measure official cache mutations without replacing their implementation."""
    meter: dict[str, float | int] = {"elapsed_ms": 0.0, "calls": 0}
    for name in (
        "update",
        "update_conv_state",
        "update_recurrent_state",
        "update_indexer",
    ):
        original = getattr(cache, name)

        def timed(*args, _original=original, **kwargs):
            synchronize(device)
            started = time.perf_counter()
            result = _original(*args, **kwargs)
            synchronize(device)
            meter["elapsed_ms"] += (time.perf_counter() - started) * 1000.0
            meter["calls"] += 1
            return result

        setattr(cache, name, timed)
    return meter


def device_check(
    device: torch.device,
    layers: list[torch.nn.Module],
    mixer: torch.nn.Module,
    lm_head: torch.Tensor,
    rotary: torch.nn.Module,
    cache: object,
    hidden: torch.Tensor | None = None,
) -> dict:
    expected_device = (
        f"cuda:{torch.cuda.current_device()}"
        if device.type == "cuda" and device.index is None
        else str(device)
    )

    def module_devices(module: torch.nn.Module) -> list[str]:
        return sorted({
            str(value.device)
            for value in list(module.parameters()) + list(module.buffers())
        })

    layer_devices = sorted({
        str(value.device)
        for layer in layers
        for value in list(layer.parameters()) + list(layer.buffers())
    })
    cache_devices = sorted({
        str(value.device)
        for layer in getattr(cache, "layers", [])
        for group in (
            "conv_states", "recurrent_states", "keys", "values",
            "indexer_keys",
        )
        for value in (
            getattr(layer, group, {}).values()
            if isinstance(getattr(layer, group, None), dict)
            else [getattr(layer, group, None)]
        )
        if value is not None
    })
    result = {
        "requested": expected_device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else None
        ),
        "layer_parameter_devices": layer_devices,
        "mixer_devices": module_devices(mixer),
        "lm_head_device": str(lm_head.device),
        "rotary_devices": module_devices(rotary),
        "cache_devices": cache_devices,
        "hidden_device": str(hidden.device) if hidden is not None else None,
    }
    result["status"] = "pass" if (
        layer_devices == [expected_device]
        and result["mixer_devices"] == [expected_device]
        and result["lm_head_device"] == expected_device
        and result["rotary_devices"] in ([], [expected_device])
        and (not cache_devices or cache_devices == [expected_device])
        and (hidden is None or result["hidden_device"] == expected_device)
    ) else "fail"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--tokens", type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expert-cache-size", type=int, default=64)
    parser.add_argument("--one-shot-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.native is None and args.tokens is None:
        raise SystemExit("provide --native or --tokens")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --device cpu")
    if args.expert_cache_size < 1:
        raise SystemExit("expert-cache-size must be positive")

    native = None
    if args.native is not None:
        native = json.loads(args.native.read_text())
        tokens = [int(step["input_token"]) for step in native.get("steps", [])]
    else:
        tokens = [int(value) for value in args.tokens.split(",") if value.strip()]
    if not tokens or any(token < 0 or token >= 248320 for token in tokens):
        raise SystemExit("input sequence contains an invalid token ID")

    device = torch.device(args.device)
    profile = Timing()
    reader = None
    meta_model = None
    result: dict
    try:
        ref = load_reference()
        config_data = json.loads((args.model_dir / "config.json").read_text())
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import (
            create_causal_mask,
            create_recurrent_attention_mask,
        )

        config = ref._qwen4.Qwen4ExpTextConfig(**config_data["text_config"])
        config._attn_implementation = "eager"
        reader = ref.GGUF(args.gguf)
        reader.set_profile(profile)
        meta_model = materialize_meta_model(ref, config)
        mixer = meta_model.hyper_connection_mixer
        mixer_prefix = "model.language_model.hyper_connection_mixer."
        for name, _ in list(mixer.named_parameters()):
            ref.replace_parameter(
                mixer, name, reader.dense(mixer_prefix + name, device)
            )
        mixer = mixer.to(device)
        lm_head = reader.dense("lm_head.weight", device)
        rotary = ref.Qwen4ExpTextRotaryEmbedding(config).to(device)
        cache = DynamicCache(config=config)
        cache_meter = instrument_cache(cache, device)
        expert_cache = ref.RoutedExpertCache(
            reader, device, args.expert_cache_size
        )

        layers = []
        print(json.dumps({
            "event": "startup",
            "message": "materializing official decoder layers on requested device",
            "device": str(device),
            "layer_count": config.num_hidden_layers,
            "expert_cache_capacity": args.expert_cache_size,
        }), flush=True)
        for layer_index in range(config.num_hidden_layers):
            layer = ref.materialize_layer(
                reader, config, layer_index, device, expert_cache
            )
            layers.append(layer)
            print(json.dumps({
                "event": "layer_ready",
                "layer": layer_index,
                "device": str(next(layer.parameters()).device),
            }), flush=True)
        setup_profile = profile.snapshot()
        initial_cache = cache_evidence(cache)
        startup_devices = device_check(
            device, layers, mixer, lm_head, rotary, cache
        )
        if startup_devices["status"] != "pass":
            raise RuntimeError(f"startup device check failed: {startup_devices}")
        print(json.dumps({
            "event": "startup_complete",
            "device_check": startup_devices,
            "setup_profile_ms": setup_profile,
        }), flush=True)

        embedding_name = "model.language_model.embed_tokens.weight"
        checkpoints: dict[str, dict] = {}
        checkpoint_layers = (0, 3, 7, 15, 31, 47)
        steps = []
        last_hidden = None

        def consume(token: int, position: int, kind: str) -> tuple[int, dict]:
            nonlocal last_hidden
            if cache.get_seq_length() != position:
                raise RuntimeError(
                    f"cache sequence length {cache.get_seq_length()} != position {position}"
                )
            before_profile = profile.snapshot()
            wall_start = time.perf_counter()
            ids = torch.tensor([[token]], dtype=torch.long, device=device)
            hidden = reader.dense_rows(
                embedding_name, [token], device
            ).reshape(1, 1, config.hidden_size)
            current_positions = torch.tensor(
                [[position]], dtype=torch.long, device=device
            )
            position_ids = current_positions[None].expand(4, -1, -1)
            text_position_ids = position_ids[0]
            rotary_positions = position_ids[1:]
            if hasattr(cache, "position_ids"):
                rotary_positions = torch.cat(
                    [cache.position_ids, rotary_positions], dim=-1
                )
            cache.position_ids = rotary_positions
            full_mask = create_causal_mask(
                config=config,
                inputs_embeds=hidden,
                attention_mask=None,
                past_key_values=cache,
                position_ids=text_position_ids,
                allow_is_causal_skip=False,
            )
            conv_mask = create_recurrent_attention_mask(
                config=config,
                inputs_embeds=hidden,
                attention_mask=None,
                past_key_values=cache,
                position_ids=text_position_ids,
            )
            if full_mask is None:
                raise RuntimeError("official causal mask unexpectedly returned None")
            ple_ids = ids
            if config.ple_layer_ids and conv_mask is not None:
                eos = config.eos_token_id
                eos = eos[0] if isinstance(eos, list) else eos
                ple_ids = torch.where(conv_mask.bool(), ple_ids, eos)
            position_embeddings = rotary(hidden, rotary_positions)
            hidden = hidden.repeat(1, 1, config.hc_count)

            forward_ms = 0.0
            cache_before_ms = float(cache_meter["elapsed_ms"])
            cache_before_calls = int(cache_meter["calls"])
            with torch.no_grad():
                for layer_index, layer in enumerate(layers):
                    synchronize(device)
                    forward_start = time.perf_counter()
                    hidden = layer(
                        hidden,
                        position_embeddings=position_embeddings,
                        attention_mask=full_mask,
                        conv_mask=conv_mask,
                        past_key_values=cache,
                        ple_input_ids=ple_ids,
                    )
                    synchronize(device)
                    forward_ms += (time.perf_counter() - forward_start) * 1000.0
                    sequence_length = cache.get_seq_length()
                    if sequence_length not in (position, position + 1):
                        raise RuntimeError(
                            "official DynamicCache changed unexpectedly after "
                            f"layer {layer_index}: {sequence_length}"
                        )
                    if layer_index in checkpoint_layers:
                        checkpoints[f"{kind}:{layer_index}"] = {
                            "kind": kind,
                            "layer": layer_index,
                            "position": position,
                            "input_token": token,
                            "stats": stats(hidden),
                        }
                if cache.get_seq_length() != position + 1:
                    raise RuntimeError(
                        "official DynamicCache did not contain the complete "
                        f"token after the decoder: {cache.get_seq_length()}"
                    )
                final_norm = mixer(hidden)
                logits = torch.matmul(
                    final_norm.float(), lm_head.float().transpose(0, 1)
                ).reshape(-1)
                next_token = int(torch.argmax(logits).item())
            last_hidden = hidden
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            device_state = device_check(
                device, layers, mixer, lm_head, rotary, cache, hidden
            )
            timing = profile.delta(before_profile)
            timing.update({
                "official_forward_ms": round(forward_ms, 3),
                "cache_update_ms": round(
                    float(cache_meter["elapsed_ms"]) - cache_before_ms, 3
                ),
                "cache_update_calls": (
                    int(cache_meter["calls"]) - cache_before_calls
                ),
                "wall_ms": round(wall_ms, 3),
                "cache_update_scope": (
                    "timed official DynamicCache update/update_conv_state/"
                    "update_recurrent_state/update_indexer calls"
                ),
            })
            evidence = {
                "kind": kind,
                "position": position,
                "input_token": token,
                "next_token": next_token,
                "prefix_length": position + 1,
                "final_norm": stats(final_norm),
                "logits": stats(logits),
                "top": top_k(logits),
                "cache_seq_length": cache.get_seq_length(),
                "timing_ms": timing,
                "device_check": device_state,
            }
            print(json.dumps({
                "event": "token_profile",
                "step": position,
                "kind": kind,
                "input_token": token,
                "next_token": next_token,
                "timing_ms": timing,
                "expert_cache": expert_cache.evidence(),
                "device_check": device_state,
            }), flush=True)
            return next_token, evidence

        current = None
        for position, token in enumerate(tokens):
            current, evidence = consume(
                token, position, "prompt" if position == 0 else "generated"
            )
            evidence["step"] = position
            steps.append(evidence)

        one_shot = None
        one_shot_path = args.one_shot_reference
        if one_shot_path is None:
            one_shot_path = Path("artifacts/m6/quant_matched_reference.json")
        if one_shot_path.exists():
            one_shot = json.loads(one_shot_path.read_text())
        comparison = {"status": "missing", "reason": "one-shot artifact absent"}
        if one_shot is not None:
            expected_logits = next(
                stage for stage in one_shot["stages"]
                if stage.get("stage") == "logits"
            )
            expected_final = next(
                stage for stage in one_shot["stages"]
                if stage.get("stage") == "final_norm"
            )
            expected_layers = {
                str(stage["layer"]): stage
                for stage in one_shot["stages"]
                if stage.get("stage") == "layer"
                and stage.get("layer") in checkpoint_layers
            }
            first = steps[0]
            checkpoint_checks = {
                str(layer): compare_stats(
                    checkpoints[f"prompt:{layer}"]["stats"],
                    expected_layers[str(layer)]["stats"],
                    2e-5,
                )
                for layer in checkpoint_layers
                if str(layer) in expected_layers
            }
            comparison = {
                "status": "pass",
                "prefix": tokens[:1],
                "position": first["position"],
                "expected_token": int(expected_logits["top"][0]["id"]),
                "actual_token": first["next_token"],
                "final_norm": compare_stats(
                    first["final_norm"], expected_final["stats"], 2e-5
                ),
                "logits": compare_stats(
                    first["logits"], expected_logits["stats"], 2e-5
                ),
                "checkpoints": checkpoint_checks,
            }
            if (
                comparison["expected_token"] != comparison["actual_token"]
                or comparison["final_norm"]["status"] != "pass"
                or comparison["logits"]["status"] != "pass"
                or len(checkpoint_checks) != len(checkpoint_layers)
                or any(item["status"] != "pass"
                       for item in checkpoint_checks.values())
            ):
                comparison["status"] = "fail"
            if comparison["status"] != "pass":
                raise RuntimeError(
                    "stateful fresh token diverges from the one-shot reference"
                )

        result = {
            "format": "q38-m6-stateful-sequence-oracle-v2",
            "reference": (
                "official Transformers Qwen4Exp decoder + official "
                "DynamicCache; independent CPU GGUF reader/dequantizer"
            ),
            "native": str(args.native) if args.native is not None else None,
            "tokens": tokens,
            "generated": [step["next_token"] for step in steps[1:]],
            "steps": steps,
            "checkpoints": list(checkpoints.values()),
            "fresh_comparison": comparison,
            "initial_cache": initial_cache,
            "final_cache": cache_evidence(cache),
            "expert_cache": expert_cache.evidence(),
            "setup_profile_ms": setup_profile,
            "device_check": device_check(
                device, layers, mixer, lm_head, rotary, cache, last_hidden
            ),
            "state_model": (
                "official DynamicCache; all non-expert layer weights and "
                "shared/router/norm/GDN/QSA/LM-head tensors resident on "
                "the requested device; bounded routed-expert CUDA LRU"
            ),
            "status": "pass",
            "reason": None,
        }
    except Exception:
        result = {
            "format": "q38-m6-stateful-sequence-oracle-v2",
            "tokens": tokens,
            "status": "blocked",
            "reason": traceback.format_exc(),
        }
    finally:
        if meta_model is not None:
            del meta_model
        if reader is not None:
            reader.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
