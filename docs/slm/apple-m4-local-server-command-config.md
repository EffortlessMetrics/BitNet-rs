# Apple M4 Local Server Command And Config Contract

Status: command/config contract plus the initial health/readiness endpoint
slice. Completion endpoints and receipt export are later work.

The Apple M4 local server should expose the already working dense SLM Mac
appliance as a loopback service while preserving the same model-cache,
tokenizer, backend, fallback, timing, memory, and receipt discipline used by
`bitnet mac ask`, `bitnet mac chat`, `bitnet mac smoke`, `bitnet mac doctor`,
and `bitnet mac regression`.

## Primary Command

The intended first-class command is:

```bash
bitnet mac serve \
  --model-id qwen2.5-0.5b-instruct-q8_0 \
  --device apple-m4-cpu-neon \
  --host 127.0.0.1 \
  --port 8080 \
  --strict \
  --stream \
  --receipt-dir ~/.local/state/bitnet-rs/receipts/apple-m4-local-server
```

`bitnet mac serve` is the Mac appliance wrapper. It should resolve the supported
model through the same cache and model matrix used by the other `bitnet mac`
commands.

The lower-level generic form may be added later, but it must preserve the same
strict fields:

```bash
bitnet serve \
  --profile apple-m4-dense-slm \
  --model-id qwen2.5-0.5b-instruct-q8_0 \
  --device apple-m4-cpu-neon \
  --host 127.0.0.1 \
  --port 8080 \
  --strict
```

## Defaults

| Field | Default | Notes |
|---|---|---|
| `model_id` | `qwen2.5-0.5b-instruct-q8_0` | The current M4 dense SLM default. |
| `device` | `apple-m4-cpu-neon` | The only full dense SLM answer backend claimed by this contract. |
| `host` | `127.0.0.1` | Loopback by default. Binding to non-loopback must be explicit. |
| `port` | `8080` | Matches the existing server crate default unless overridden. |
| `strict` | `true` | Hidden fallback is not allowed. |
| `stream` | `true` | Token streaming should be the default user experience. |
| `cache_dir` | existing Mac model cache default | Override with `--cache-dir` or config. |
| `receipt_dir` | local user state receipt directory | Override with `--receipt-dir`. |
| `receipt_mode` | `per_request` | Aggregate session receipts can be added later. |

The supported non-default dense model can be selected explicitly:

```bash
bitnet mac serve \
  --model-id qwen2.5-0.5b-instruct-q4_k_m \
  --device apple-m4-cpu-neon \
  --strict
```

## Config File Shape

The server should accept an optional config file equivalent to the command-line
contract:

```toml
[server]
host = "127.0.0.1"
port = 8080
stream = true

[model]
model_id = "qwen2.5-0.5b-instruct-q8_0"
cache_dir = "~/.cache/bitnet-rs/models"
strict_cache = true
offline = false

[runtime]
device = "apple-m4-cpu-neon"
strict_loader = true
strict_tokenizer = true
hidden_fallback_allowed = false

[receipts]
enabled = true
mode = "per_request"
dir = "~/.local/state/bitnet-rs/receipts/apple-m4-local-server"
include_generated_text = true
include_token_ids = true
include_timing = true
include_memory = true
```

CLI flags should override config-file values. Startup must print or record the
resolved config in the server receipt context without leaking prompt content
outside request receipts.

## Startup Checks

`bitnet mac serve` startup should be cheap enough for normal local use while
still rejecting unsafe states before listening:

- resolve `model_id` through the supported M4 dense model matrix;
- verify cache metadata and model SHA before accepting requests;
- verify tokenizer authority and prompt template;
- confirm `requested_backend = apple-m4-cpu-neon`;
- confirm `selected_backend = apple-m4-cpu-neon`;
- confirm `fallback_used = false`;
- reject unsupported full `apple-m4-metal` server requests until a later
  full-route receipt proves that backend;
- report available disk and cache state in readiness output;
- never download a model unless a future explicit `--fetch-if-missing` flag is
  added and documented.

Cache verification may reuse already verified cache metadata for request-time
latency, but startup must distinguish `sha256_from_metadata` from a fresh file
rehash in receipts.

## Failure Behavior

Failures must be explicit and operator-actionable:

| Condition | Required behavior |
|---|---|
| Missing model cache | Refuse to start and suggest `bitnet model fetch <model_id>`. |
| Wrong model hash | Refuse to start and suggest `bitnet model prune <model_id>` then fetch. |
| Missing tokenizer authority | Refuse to start; do not fall back to a guessed tokenizer. |
| Unsupported model id | Refuse to start and show supported model IDs. |
| `--device apple-m4-metal` | Refuse full-server mode until full route support is proven. |
| Hidden fallback would occur | Refuse to start or reject the request with a fallback error. |
| Non-loopback host | Allow only when explicitly configured; warn that this is local-service scope. |
| Receipt directory unwritable | Refuse to start unless `--receipt-mode off` is explicitly supported later. |

## Endpoint Contract For Later Items

`M4-SERVE-001` does not implement endpoints. Later items should use this
contract:

| Endpoint | First item | Purpose |
|---|---|---|
| `GET /health` | `M4-SERVE-002` | Implemented as process health and cheap server status. |
| `GET /health/live` | `M4-SERVE-002` | Implemented as a liveness alias for `/health`. |
| `GET /ready` | `M4-SERVE-002` | Implemented with model-cache, tokenizer, backend, fallback, disk, and receipt readiness. |
| `GET /health/ready` | `M4-SERVE-002` | Implemented as a readiness alias for `/ready`. |
| `POST /v1/chat/completions` | `M4-SERVE-003` | Streaming dense SLM completion surface. |
| `GET /receipts/{id}` | `M4-SERVE-004` | Export strict per-request receipts. |

The completion endpoint may be OpenAI-shaped, but full OpenAI compatibility must
not be claimed until request/response semantics, streaming chunks, errors, and
receipts are tested.

`M4-SERVE-002` does not run generation. Readiness reports whether startup
verified the supported model cache, tokenizer authority, `apple-m4-cpu-neon`
backend route, no-hidden-fallback policy, disk/cache state, and receipt
directory. Missing or invalid cache still prevents startup.

## Receipt Requirements

Every completed generation request should be able to export a receipt with:

- `artifact_kind = "bitnet_apple_m4_local_server_request"`;
- `server.host`, `server.port`, `server.endpoint`, and `server.request_id`;
- model ID, source, size, SHA256, and SHA256 source;
- tokenizer authority and prompt template;
- requested backend, selected backend, runtime API, and fallback status;
- generated text and generated token IDs;
- prompt token IDs when enabled by receipt policy;
- time to first token, decode timing, total request timing, and memory;
- streaming enabled/disabled status;
- cache verification status;
- claim-boundary fields stating that dense SLM server success does not prove
  BitNet, QK256, Neural Engine, MPSGraph, full Metal inference, or broad M4
  performance.

Failed startup and failed request receipts should record the failing gate when a
receipt directory is available.

## Claim Boundary

This contract may claim only that the M4 local server command/config surface and
initial health/readiness endpoint slice are defined.

It must not claim:

- a generation endpoint is implemented;
- streaming completions work;
- OpenAI compatibility is proven;
- production deployment readiness;
- BitNet local-answer quality;
- full `apple-m4-metal` inference;
- Neural Engine execution;
- MPSGraph model inference;
- QK256 support;
- broad M4 performance.
