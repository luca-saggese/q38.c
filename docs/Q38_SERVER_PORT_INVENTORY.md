# Q38 Full Server Port Inventory

This document records the server port from the repository `main` branch and
the adaptation boundary for the Q38 runtime. The donor snapshot inspected for
this port is `antirez/ds4` commit `9ab705347c1775e7599ede7eb81a6255ec7dccb5`,
which is also available as the local `main` branch.

The port preserves the donor's protocol behavior and test intent, but does
not import the donor inference engine or its model-specific prompt syntax.
The direct `q38` binary and its `q38_runtime -> q38_session` execution path
remain independent of the HTTP server.

## Donor surface

| Donor | Responsibility | Q38 destination |
| --- | --- | --- |
| `ds4_server.c` | HTTP parsing, request validation, protocol response mapping, SSE, queueing, slots, cancellation, logging, tracing, and server tests | `q38_server.c` plus protocol modules |
| `ds4_web.c/h` | Optional browser/CDP web helper | `q38_web.c/h` only where the server-facing feature is required |
| `ds4_kvstore.c/h` | Disk prefix/session cache plumbing | `q38_kvstore.c/h` with Q38 session hooks |
| `ds4_help.c/h` | CLI help text | Q38-specific help output |
| `tests/ds4_test.c` | Protocol and server unit-test blocks mixed with engine tests | `tests/q38_server_test.c` and model-free fixtures |

The donor server is approximately 18k lines and contains the complete
protocol implementation. It is not a license to copy donor engine internals:
the only runtime boundary to the model is `q38_server_engine`.

## Protocol features to retain

The Q38 server must retain:

- HTTP/1.1 parsing, request-size limits, connection handling, CORS, logging,
  trace IDs, disconnect detection, and cancellation;
- `/v1/models`, `/v1/completions`, `/v1/chat/completions`,
  `/v1/responses`, and `/v1/messages`;
- streaming and non-streaming responses, SSE keepalive during prefill, final
  usage, and optional streamed usage;
- OpenAI reasoning content, Responses reasoning items, and Anthropic thinking
  blocks;
- function, custom, and namespace tool schemas, streamed tool calls, tool
  results, continuation IDs, live continuation state, and invalid-call
  recovery plumbing;
- image URL, data URI, base64, PNG, and JPEG request representation;
- resident slots, request queueing, session ownership, cancellation, and
  cache read/write hints;
- model-free protocol tests for every retained surface.

## Q38-specific replacement boundary

The server protocol layer must not depend on Qwen tensor or CUDA details. It
calls only the following engine boundary:

```text
q38_server_engine
  -> q38_runtime
  -> q38_session
  -> q38_session_prefill / q38_session_prefill_chunked
  -> q38_session_eval_timed
  -> q38_session_emit / q38_session_stream_token
```

The prompt adapter owns:

- system/user/assistant rendering;
- Qwen3.8 reasoning representation;
- Qwen3.8 tool schema and tool-result rendering;
- tool-call extraction from Qwen output;
- multimodal placeholders and special-token handling.

The final Q38 runtime must not contain donor DeepSeek/GLM model selection,
DeepSeek or GLM templates, DSML delimiters, donor model tool markers, or
donor vision encoder calls. Protocol structures may retain generic
`tool_call`, `reasoning`, `image`, and `tool_result` concepts.

## Import phases

| Phase | State | Constraint |
| --- | --- | --- |
| SERVER-00 donor inventory | complete | Static inspection only |
| SERVER-01 HTTP/protocol types | complete | No model load |
| SERVER-02 mock engine | complete | All protocol tests use mock behavior |
| SERVER-03 Qwen prompt adapter | complete | No donor model markers; extraction fixture added |
| SERVER-04 engine boundary | complete | Vtable boundary plus real adapter source |
| SERVER-05 resident server | pending | One runtime load at server startup |
| SERVER-06 `q38-cli` | complete | Must not link GGUF/CUDA |
| SERVER-07 server protocol tests | complete | `make test-server` is model-free |
| SERVER-08 real acceptance | pending | At most one fresh model load |

No real model load is part of SERVER-00 through SERVER-07.

## Attribution

The protocol implementation is mechanically adapted from the MIT-licensed
`antirez/ds4` server sources. Adapted files must retain an attribution note
and the repository MIT license. Q38-specific engine, prompt, and runtime
code remains separate from donor code so future donor updates can be merged
without reintroducing model-specific assumptions.
