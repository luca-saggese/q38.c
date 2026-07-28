#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

MODEL="${DS4_MODEL:-/home/antirez/models/deepseek-v4-gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf}"
CTX="${DS4_CTX:-100000}"
SERVER_HOST="${DS4_SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${DS4_SERVER_PORT:-8000}"
KV_DIR="${DS4_KV_DIR:-/data/ds4-kv}"
KV_SPACE_MB="${DS4_KV_SPACE_MB:-8192}"
BATCHED_SESSIONS="${DS4_BATCHED_SESSIONS:-16}"

if [[ ! -x ./ds4-server ]]; then
    printf 'run-nvidia-tp-server: ./ds4-server is not built; run `make cuda-generic` first\n' >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    printf 'run-nvidia-tp-server: Q4 model not found: %s\n' "$MODEL" >&2
    exit 1
fi

exec ./ds4-server \
    --cuda \
    --cuda-tensor-parallel \
    --gpu-devices 0,2,4,6,1,3,5,7 \
    --gpu-vram auto \
    --model "$MODEL" \
    --ctx "$CTX" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --batched-session "$BATCHED_SESSIONS" \
    --kv-disk-dir "$KV_DIR" \
    --kv-disk-space-mb "$KV_SPACE_MB" \
    "$@"
