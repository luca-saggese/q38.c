#!/usr/bin/env python3
"""Cache-aware, quant-matched Qwen4Exp oracle.

The decoder equations and cache mutations are supplied by the installed
Transformers Qwen4Exp implementation.  GGUF tensors are materialized for one
decoder layer at a time; only activations, the official cache, and the current
layer remain live across a token step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path

import torch


def load_reference():
    path = Path(__file__).with_name("m6_quant_matched_reference.py")
    spec = importlib.util.spec_from_file_location("q38_m6_reference", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tensor_evidence(value: torch.Tensor | None) -> dict:
    if value is None:
        return {"present": False}
    detached = value.detach().contiguous()
    float_value = detached.float()
    flat = float_value.reshape(-1).cpu()
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    raw = float_value.cpu().numpy().tobytes()
    evidence = {
        "present": True,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": bool(finite.all().item()),
        "finite_count": int(finite.sum()),
        "nan_count": int(torch.isnan(flat).sum()),
        "inf_count": int(torch.isinf(flat).sum()),
        "checksum": hashlib.sha256(raw).hexdigest(),
    }
    if finite_values.numel():
        evidence.update({
            "min": float(finite_values.min()),
            "max": float(finite_values.max()),
            "mean": float(finite_values.mean()),
            "rms": float(torch.sqrt(torch.mean(finite_values * finite_values))),
            "max_abs": float(torch.max(torch.abs(finite_values))),
        })
    else:
        evidence.update({
            "min": None, "max": None, "mean": None, "rms": None,
            "max_abs": None,
        })
    if flat.numel() <= 16:
        evidence["values"] = [float(item) for item in flat.tolist()]
    return evidence


def cache_evidence(cache: DynamicCache) -> dict:
    result = {"position_ids": tensor_evidence(getattr(cache, "position_ids", None)),
              "layers": []}
    for index, layer in enumerate(cache.layers):
        entry = {
            "layer": index,
            "type": type(layer).__name__,
            "is_initialized": bool(getattr(layer, "is_initialized", False)),
            "has_previous_state": {
                str(key): bool(value)
                for key, value in getattr(layer, "has_previous_state", {}).items()
            },
            "conv_states": {},
            "recurrent_states": {},
        }
        for key, value in getattr(layer, "conv_states", {}).items():
            entry["conv_states"][str(key)] = tensor_evidence(value)
        for key, value in getattr(layer, "recurrent_states", {}).items():
            entry["recurrent_states"][str(key)] = tensor_evidence(value)
        entry["keys"] = tensor_evidence(getattr(layer, "keys", None))
        entry["values"] = tensor_evidence(getattr(layer, "values", None))
        entry["indexer_keys"] = tensor_evidence(
            getattr(layer, "indexer_keys", None)
        )
        entry["is_indexer_initialized"] = bool(
            getattr(layer, "is_indexer_initialized", False)
        )
        result["layers"].append(entry)
    return result


def device_check(
    device: torch.device,
    mixer: torch.nn.Module,
    lm_head: torch.Tensor,
    rotary: torch.nn.Module,
    cache: DynamicCache,
    hidden: torch.Tensor,
    layer: torch.nn.Module | None = None,
) -> dict:
    expected = (
        f"cuda:{torch.cuda.current_device()}"
        if device.type == "cuda" and device.index is None
        else str(device)
    )

    def module_devices(module: torch.nn.Module) -> list[str]:
        return sorted({
            str(value.device)
            for value in list(module.parameters()) + list(module.buffers())
        })

    layer_devices = module_devices(layer) if layer is not None else []
    cache_devices = sorted({
        str(value.device)
        for cached_layer in cache.layers
        for group in (
            "conv_states", "recurrent_states", "keys", "values",
            "indexer_keys",
        )
        for value in (
            getattr(cached_layer, group, {}).values()
            if isinstance(getattr(cached_layer, group, None), dict)
            else [getattr(cached_layer, group, None)]
        )
        if value is not None
    })
    result = {
        "requested": expected,
        "layer_devices": layer_devices,
        "mixer_devices": module_devices(mixer),
        "lm_head_device": str(lm_head.device),
        "rotary_devices": module_devices(rotary),
        "cache_devices": cache_devices,
        "hidden_device": str(hidden.device),
    }
    result["status"] = "pass" if (
        (not layer_devices or layer_devices == [expected])
        and result["mixer_devices"] == [expected]
        and result["lm_head_device"] == expected
        and result["rotary_devices"] in ([], [expected])
        and (not cache_devices or cache_devices == [expected])
        and result["hidden_device"] == expected
    ) else "fail"
    return result


def stats(value: torch.Tensor) -> dict:
    flat = value.detach().float().reshape(-1).cpu()
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    raw = flat.numpy().tobytes()
    if finite_values.numel():
        return {
            "min": float(finite_values.min()),
            "max": float(finite_values.max()),
            "mean": float(finite_values.mean()),
            "rms": float(torch.sqrt(torch.mean(finite_values * finite_values))),
            "max_abs": float(torch.max(torch.abs(finite_values))),
            "finite_count": int(finite.sum()),
            "nan_count": int(torch.isnan(flat).sum()),
            "inf_count": int(torch.isinf(flat).sum()),
            "checksum": hashlib.sha256(raw).hexdigest(),
            "fixed": [
                {"index": i, "value": float(flat[i]) if i < flat.numel() else None}
                for i in range(4)
            ],
        }
    return {
        "min": None,
        "max": None,
        "mean": None,
        "rms": None,
        "max_abs": None,
        "finite_count": 0,
        "nan_count": int(torch.isnan(flat).sum()),
        "inf_count": int(torch.isinf(flat).sum()),
        "checksum": hashlib.sha256(raw).hexdigest(),
        "fixed": [],
    }


def top_k(value: torch.Tensor, count: int = 10) -> list[dict]:
    flat = value.detach().float().reshape(-1).cpu().tolist()
    order = sorted(range(len(flat)), key=lambda i: (-flat[i], i))[:count]
    return [{"id": i, "value": flat[i]} for i in order]


def compare_stats(left: dict, right: dict, tolerance: float = 1e-5) -> dict:
    fields = ("min", "max", "mean", "rms", "max_abs")
    deltas = {}
    for field in fields:
        a, b = left.get(field), right.get(field)
        if a is None or b is None:
            deltas[field] = None
        else:
            deltas[field] = abs(float(a) - float(b))
    failed = [
        field for field, delta in deltas.items()
        if delta is not None and delta > tolerance
    ]
    return {
        "status": "pass" if not failed else "fail",
        "tolerance": tolerance,
        "deltas": deltas,
        "failed_fields": failed,
        "checksum_equal": left.get("checksum") == right.get("checksum"),
    }


def materialize_meta_model(ref, config):
    """Construct the official graph on meta, without allocating model weights."""
    with torch.device("meta"):
        model = ref._qwen4.Qwen4ExpTextModel(config)
    if any(not parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("meta model unexpectedly contains materialized weights")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="9419")
    parser.add_argument("--generated-count", type=int, required=True)
    parser.add_argument(
        "--fresh-only",
        action="store_true",
        help="stop after the fresh prompt step and its C12 comparison",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-layer", type=int)
    parser.add_argument(
        "--one-shot-reference",
        type=Path,
        default=Path("artifacts/m6/quant_matched_reference.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.generated_count < 1 or args.generated_count > 128:
        raise SystemExit("generated-count must be in [1,128]")
    prompt = [int(value) for value in args.prompt.split(",") if value.strip()]
    if not prompt or any(value < 0 or value >= 248320 for value in prompt):
        raise SystemExit("prompt contains an invalid token ID")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --device cpu")

    device = torch.device(args.device)
    try:
        ref = load_reference()
        config_data = json.loads((args.model_dir / "config.json").read_text())
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import (
            create_causal_mask,
            create_recurrent_attention_mask,
        )
    except Exception:
        result = {
            "format": "q38-m6-stateful-gguf-oracle-v2",
            "prompt": prompt,
            "status": "blocked",
            "reason": traceback.format_exc(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        raise SystemExit(1)
    config = ref._qwen4.Qwen4ExpTextConfig(**config_data["text_config"])
    # Standalone official attention modules require an explicit eager interface.
    config._attn_implementation = "eager"
    reader = ref.GGUF(args.gguf)
    meta_model = None
    try:
        meta_model = materialize_meta_model(ref, config)
        rotary = ref.Qwen4ExpTextRotaryEmbedding(config).to(device)
        mixer = meta_model.hyper_connection_mixer
        mixer_prefix = "model.language_model.hyper_connection_mixer."
        for name, _ in list(mixer.named_parameters()):
            ref.replace_parameter(
                mixer, name, reader.dense(mixer_prefix + name, device)
            )
        mixer = mixer.to(device)
        lm_head = reader.dense("lm_head.weight", device)
        cache = DynamicCache(config=config)
        initial_cache = cache_evidence(cache)
        embedding_name = "model.language_model.embed_tokens.weight"
        layer_count = (
            config.num_hidden_layers
            if args.max_layer is None
            else min(config.num_hidden_layers, args.max_layer + 1)
        )
        if layer_count != config.num_hidden_layers:
            raise RuntimeError(
                "max-layer is diagnostic only; full-model oracle requires all layers"
            )

        steps = []
        checkpoints = {}

        def consume(token: int, position: int, kind: str) -> tuple[int, dict]:
            nonlocal cache
            if cache.get_seq_length() != position:
                raise RuntimeError(
                    f"cache sequence length {cache.get_seq_length()} != position {position}"
                )
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
            with torch.no_grad():
                for layer_index in range(config.num_hidden_layers):
                    layer = ref.materialize_layer(
                        reader, config, layer_index, device
                    )
                    try:
                        layer_device_check = device_check(
                            device, mixer, lm_head, rotary, cache, hidden, layer
                        )
                        if layer_device_check["status"] != "pass":
                            raise RuntimeError(
                                f"layer {layer_index} device check failed: "
                                f"{layer_device_check}"
                            )
                        hidden = layer(
                            hidden,
                            position_embeddings=position_embeddings,
                            attention_mask=full_mask,
                            conv_mask=conv_mask,
                            past_key_values=cache,
                            ple_input_ids=ple_ids,
                        )
                    finally:
                        del layer
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    if layer_index in (0, 3, 7, 15, 31, 47):
                        checkpoints[f"{kind}:{layer_index}"] = {
                            "kind": kind,
                            "layer": layer_index,
                            "position": position,
                            "input_token": token,
                            "stats": stats(hidden),
                        }
                final_norm = mixer(hidden)
                logits = torch.matmul(
                    final_norm.float(), lm_head.float().transpose(0, 1)
                ).reshape(-1)
                next_token = int(torch.argmax(logits).item())
            evidence = {
                "kind": kind,
                "position": position,
                "input_token": token,
                "next_token": next_token,
                "prefix_length": position + 1,
                "final_norm": stats(final_norm),
                "logits": stats(logits),
                "top": top_k(logits),
                "cache": cache_evidence(cache),
                "state_trace": {
                    "committed_position": position,
                    "input_token": token,
                    "cache_seq_length": cache.get_seq_length(),
                    "device_check": device_check(
                        device, mixer, lm_head, rotary, cache, hidden
                    ),
                },
            }
            if evidence["state_trace"]["device_check"]["status"] != "pass":
                raise RuntimeError(
                    f"state trace device check failed: "
                    f"{evidence['state_trace']['device_check']}"
                )
            return next_token, evidence

        current = None
        position = 0
        prompt_steps = []
        for token in prompt:
            current, evidence = consume(token, position, "prompt")
            prompt_steps.append(evidence)
            position += 1

        # The fresh-session step is compared before any continuation is run.
        one_shot = None
        fresh = prompt_steps[0] if len(prompt) == 1 else None
        if args.one_shot_reference.exists():
            one_shot = json.loads(args.one_shot_reference.read_text())
        fresh_comparison = {"status": "missing"}
        if fresh is not None and one_shot is not None:
            one_shot_logits = next(
                stage for stage in one_shot.get("stages", [])
                if stage.get("stage") == "logits"
            )
            one_shot_layers = {
                stage["layer"]: stage
                for stage in one_shot.get("stages", [])
                if stage.get("stage") == "layer"
                and stage.get("layer") in (0, 3, 7, 15, 31, 47)
            }
            checkpoint_comparison = {}
            for checkpoint in (
                item for item in checkpoints.values()
                if item["kind"] == "prompt"
            ):
                layer = checkpoint["layer"]
                expected = one_shot_layers.get(layer)
                checkpoint_comparison[str(layer)] = (
                    {"status": "missing"}
                    if expected is None
                    else compare_stats(
                        checkpoint["stats"], expected["stats"], tolerance=2e-5
                    )
                )
            expected_token = int(one_shot_logits["top"][0]["id"])
            fresh_comparison = {
                "status": "pass"
                if fresh["next_token"] == expected_token
                else "fail",
                "expected_token": expected_token,
                "actual_token": fresh["next_token"],
                "prefix": prompt,
                "position": fresh["position"],
                "final_norm": compare_stats(
                    fresh["final_norm"],
                    next(
                        stage for stage in one_shot["stages"]
                        if stage.get("stage") == "final_norm"
                    )["stats"],
                    tolerance=2e-5,
                ),
                "logits": compare_stats(
                    fresh["logits"], one_shot_logits["stats"], tolerance=2e-5
                ),
                "checkpoints": checkpoint_comparison,
            }
            if (
                fresh_comparison["final_norm"]["status"] != "pass"
                or fresh_comparison["logits"]["status"] != "pass"
                or any(
                    item["status"] != "pass"
                    for item in checkpoint_comparison.values()
                )
            ):
                fresh_comparison["status"] = "fail"
            if fresh_comparison["status"] != "pass":
                raise RuntimeError(
                    "fresh-session stateful result diverges from C12 one-shot"
                )

        generated = []
        continuation_steps = []
        if not args.fresh_only:
            for step in range(args.generated_count):
                current, evidence = consume(current, position, "generated")
                evidence["step"] = step
                generated.append(current)
                continuation_steps.append(evidence)
                position += 1

        result = {
            "format": "q38-m6-stateful-gguf-oracle-v2",
            "reference": (
                "official Transformers Qwen4Exp decoder/cache + independent "
                "GGUF reader/dequantizer"
            ),
            "transformers_version": config_data["text_config"].get(
                "transformers_version"
            ),
            "gguf": str(args.gguf),
            "prompt": prompt,
            "fresh_session": {
                "status": "pass" if fresh_comparison.get("status") == "pass"
                else "not_compared",
                "step": fresh,
                "comparison": fresh_comparison,
                "initial_cache": initial_cache,
            },
            "prompt_steps": prompt_steps,
            "generated": generated,
            "steps": continuation_steps,
            "checkpoints": list(checkpoints.values()),
            "final_cache": cache_evidence(cache),
            "state_model": (
                "official DynamicCache; one decoder layer materialized and "
                "released per layer/token"
            ),
            "status": "pass",
            "reason": None,
        }
    except Exception as exc:
        result = {
            "format": "q38-m6-stateful-gguf-oracle-v2",
            "prompt": prompt,
            "status": "blocked",
            "reason": traceback.format_exc(),
        }
    finally:
        if meta_model is not None:
            del meta_model
        reader.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
