# Native M8 R1 quality records

`tests/m8_r1_quality` is a focused recorder for the R1 runtime artifact. It
reads the same line-oriented corpus used by the Transformers reference and
writes one `q38_eval.py`-compatible JSON record per sequence:

```sh
make tests/m8_r1_quality
./tests/m8_r1_quality \
  artifacts/m8/qwen38-runtime-R1-Q4Experts-BF16Core-BF16PLE.gguf \
  artifacts/m8/quality_corpus.jsonl > artifacts/m8/R1_quality_records.jsonl
```

Records contain the final-token full logits, target logit, top-20 logits,
per-layer top-10 router IDs/weights, and per-layer QSA selections. The
callbacks exposed by `q38_forward_full` do not include a token index, so for
multi-token sequences the route and QSA fields intentionally record the final
token in each layer; this is the hard limitation of the current native trace
API. No vectors are synthesized and PLE execution is unchanged.
