# Q38 Directional Steering

Directional steering is an optional activation edit for Qwen3.8. A direction
file is a flat FP32 matrix with one normalized `2560`-wide vector for each of
the `48` Q38 layers:

```text
48 * 2560 float32 = 1,966,080 bytes
y = y - scale * direction[layer] * dot(direction[layer], y)
```

Positive scales suppress the represented direction and negative scales amplify
it. With no file, or with both scales set to zero, the normal inference path is
unchanged.

## Runtime options

```text
--dir-steering-file FILE
--dir-steering-ffn FLOAT
--dir-steering-attn FLOAT
```

The default when a file is supplied without an explicit scale is
`--dir-steering-ffn 1`. Attention steering applies after the QSA output
projection and the GDN mixer output. FFN steering applies after the complete
MoE/shared-FFN output. Request-scoped server overrides use:

```json
{"q38":{"steering":{"ffn":-0.5,"attn":0.0}}}
```

The server default is never mutated by a request override.

## Building a direction

The extractor uses the direct Q38 binary and its activation dump mode; it does
not create a second inference path:

```sh
python3 dir-steering/tools/build_direction.py \
  --q38 ./q38 \
  --model model.gguf \
  --tokenizer /path/to/tokenizer \
  --good-file dir-steering/examples/succinct.txt \
  --bad-file dir-steering/examples/verbose.txt \
  --out dir-steering/out/verbosity.json \
  --component ffn_out
```

The resulting `.f32` file is validated as `48x2560`, finite, and
per-layer-normalized before it is written.

## Scale sweep

```sh
python3 dir-steering/tools/run_sweep.py \
  --q38 ./q38 \
  --model model.gguf \
  --tokenizer /path/to/tokenizer \
  --direction dir-steering/out/verbosity.f32 \
  --prompts dir-steering/examples/eval_prompts.txt \
  --scales "-1,-0.5,0,0.5,1,2"
```

Reference 0 is always steering-disabled. Steering experiments belong in a
separate benchmark suite and must record the direction fingerprint and both
effective scales.
