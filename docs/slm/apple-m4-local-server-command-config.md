# Apple M4 Local Server Command And Config Contract

Status: command/config contract, health/readiness endpoints, model catalog
endpoint, the first streaming local completion endpoint, HTTP export for
per-request receipts, and operator readiness checking.

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
  --receipt-dir ~/.local/state/bitnet-rs/receipts/apple-m4-local-server \
  --trace
```

`bitnet mac serve` is the Mac appliance wrapper. It should resolve the supported
model through the same cache and model matrix used by the other `bitnet mac`
commands. Startup prints the loopback URL and the primary operator endpoints:
`/health`, `/models`, and `/ready`.

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
| `allow_network_bind` | `false` | Pass `--allow-network-bind` with a non-loopback `--host`; otherwise startup fails before cache lookup or bind. |
| `port` | `8080` | Matches the existing server crate default unless overridden. |
| `strict` | `true` | Hidden fallback is not allowed. |
| `stream` | `true` | Token streaming should be the default user experience. |
| `cache_dir` | existing Mac model cache default | Override with `--cache-dir` or config. |
| `receipt_dir` | local user state receipt directory | Override with `--receipt-dir`. |
| `receipt_mode` | `per_request` | Aggregate session receipts can be added later. |
| `trace` | `false` | Override with `--trace` to emit an `m4obs-*` correlation ID in redacted diagnostics and receipts. |

## Safety Defaults

`bitnet mac serve` is a local appliance wrapper first. It binds to
`127.0.0.1` by default and refuses `0.0.0.0`, LAN IPs, or other non-loopback
hosts unless the operator passes `--allow-network-bind`. That opt-in is still
not a production-hosting claim; it only acknowledges that local server metadata
may be reachable outside the machine.

The health, ready, models, completion-response, and receipt-export surfaces
publish a `safety` or redaction policy with these defaults:

- telemetry is disabled; there is no metrics upload or external operator
  metrics endpoint;
- CORS is disabled by default: no wildcard origin, no credentialed browser
  access, and no preflight route;
- completion request bodies are capped at 1 MiB and oversized requests return
  `413 Payload Too Large` before generation;
- operator HTTP metadata redacts cache roots, cache paths, metadata paths,
  symlink targets, model paths, and receipt directories;
- `GET /receipts/{id}` accepts only a single safe receipt file stem and exports
  a cache-path-redacted view of the local receipt file.

Local receipt files may still contain prompt text, token IDs, generated text,
and local artifact paths because they are the evidence artifact for the run.
Trace output remains redacted for prompt text and cache paths.

The supported non-default dense model can be selected explicitly:

```bash
bitnet mac serve \
  --model-id qwen2.5-0.5b-instruct-q4_k_m \
  --device apple-m4-cpu-neon \
  --strict
```

The larger supported Qwen-class model is also explicit-only:

```bash
bitnet mac serve \
  --model-id qwen2.5-1.5b-instruct-q4_k_m \
  --device apple-m4-cpu-neon \
  --strict
```

Use this only after `bitnet model fetch` / `bitnet model verify` succeeds for
that model ID. It has a larger cache footprint and currently has bounded
quality/registration evidence, not a published release-mode server envelope.

Operators can verify a running server without starting a second model process:

```bash
bitnet mac serve-check --url http://127.0.0.1:8080
```

For an end-to-end local service smoke that exercises readiness, one completion,
and receipt export:

```bash
bitnet mac serve-check \
  --url http://127.0.0.1:8080 \
  --completion \
  --max-new-tokens 1
```

`serve-check` writes a compact operator receipt and does not claim production
uptime, full OpenAI compatibility, BitNet quality, or full Metal inference. The
default check verifies `/ready` and `/models`; `--completion` additionally
checks one completion and receipt export. The `/models` check records the
recommended first model ID plus exact fetch/verify commands when disk headroom
allows a supported model fetch.

For the M4 excellence server refresh, use the dense-only in-process smoke to
exercise the same local server handlers without enabling BitNet serve:

```bash
bitnet --device apple-m4-cpu-neon mac serve-smoke \
  --model-id qwen2.5-0.5b-instruct-q8_0 \
  --receipt-dir ci/hardware/apple-m4-mac-mini/<date>/slm-serve/qwen2.5-0.5b-instruct-q8_0/receipts \
  --json-out ci/hardware/apple-m4-mac-mini/<date>/slm-serve/qwen2.5-0.5b-instruct-q8_0/serve-smoke.json
```

`serve-smoke` verifies health, ready, models, non-streaming completion,
streaming completion, per-request receipt export, backend/fallback fields, and
claim boundaries. It remains a local conformance receipt, not production
hosting, broad OpenAI compatibility, BitNet serve readiness, or Metal evidence.

For the follow-on semantics proof, use the model-free failure-semantics smoke:

```bash
bitnet --device apple-m4-cpu-neon mac serve-failure-smoke \
  --json-out ci/hardware/apple-m4-mac-mini/<date>/serve-failure-semantics/summary.json
```

`serve-failure-smoke` records the bounded `M4-SERVE-EX-002` contract for dense
SLM and gated BitNet serve routes: partial token streaming, client
cancellation, timeout stage, invalid request, missing cache, per-request
receipt export, and no-response failure receipts. The receipt is generic
PR/CI-safe and does not run or download models; live dense runtime coverage
comes from `M4-SERVE-EX-001`, while BitNet serve remains tied to the
`M4-BITNET-EX-007` gate. It does not claim production hosting, full OpenAI
compatibility, full Metal inference, broad quality, or broad performance.

## M4-SERVE-EX-001 Dense Server Refresh

The 2026-05-20 M4 dense server refresh ran the in-process `serve-smoke`
against all supported dense SLM model identities:

| Model ID | Aggregate receipt | One-shot completion | Streaming completion | Backend | Fallback |
|---|---|---:|---:|---|---|
| `qwen2.5-0.5b-instruct-q8_0` | `ci/hardware/apple-m4-mac-mini/2026-05-20T2147Z/slm-serve/qwen2.5-0.5b-instruct-q8_0/serve-smoke.json` | pass, 8 tokens | pass, 8 tokens | `apple-m4-cpu-neon` | `false` |
| `qwen2.5-0.5b-instruct-q4_k_m` | `ci/hardware/apple-m4-mac-mini/2026-05-20T2147Z/slm-serve/qwen2.5-0.5b-instruct-q4_k_m/serve-smoke.json` | pass, 8 tokens | pass, 8 tokens | `apple-m4-cpu-neon` | `false` |
| `qwen2.5-1.5b-instruct-q4_k_m` | `ci/hardware/apple-m4-mac-mini/2026-05-20T2147Z/slm-serve/qwen2.5-1.5b-instruct-q4_k_m/serve-smoke.json` | pass, 8 tokens | pass, 8 tokens | `apple-m4-cpu-neon` | `false` |

Each aggregate receipt validates as
`bitnet_apple_m4_dense_local_server_smoke` and includes health, ready, models,
non-streaming completion, streaming completion, per-request receipt export,
backend/fallback fields, and claim boundaries. The in-process smoke records the
timeout boundary as metadata-only with enforcement deferred to
`M4-SERVE-EX-002`; it does not claim production hosting, full OpenAI
compatibility, BitNet serve, full Metal inference, QK256, Neural Engine,
MPSGraph, speedup, broad model quality, broad performance, or broad Apple
Silicon behavior.

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
outside request receipts. When `--trace` is enabled, startup diagnostics,
completion responses, and per-request receipts share a trace ID while prompt
text, secret values, and cache paths remain redacted from trace output.

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
| Non-loopback host | Refuse unless `--allow-network-bind` is passed; warn that this is local-service scope. |
| Receipt directory unwritable | Refuse to start unless `--receipt-mode off` is explicitly supported later. |

## Endpoint Contract

The current local server surface is:

| Endpoint | First item | Purpose |
|---|---|---|
| `GET /health` | `M4-SERVE-002` | Implemented as process health and cheap server status. |
| `GET /health/live` | `M4-SERVE-002` | Implemented as a liveness alias for `/health`. |
| `GET /models` | `M4-UX-001` | Implemented as a no-generation, no-download model catalog with cache and disk guidance. |
| `GET /ready` | `M4-SERVE-002` | Implemented with model-cache, tokenizer, backend, fallback, disk, and receipt readiness. |
| `GET /health/ready` | `M4-SERVE-002` | Implemented as a readiness alias for `/ready`. |
| `POST /v1/chat/completions` | `M4-SERVE-003` | Implemented as a streaming dense SLM completion surface with per-request receipts. |
| `GET /receipts/{id}` | `M4-SERVE-004` | Implemented to export strict per-request receipts from the configured receipt directory. |

The completion endpoint may be OpenAI-shaped, but full OpenAI compatibility must
not be claimed until request/response semantics, streaming chunks, errors, and
receipts are tested.

`M4-SERVE-003` accepts a narrow OpenAI-shaped request with either `prompt` or a
`messages` array, optional `max_tokens`/`max_new_tokens`, sampling defaults, and
`stream`. The server loads the supported model/tokenizer at startup, serializes
completion requests through that resident state, and writes a strict per-request
receipt under `receipt_dir`. `M4-SERVE-004` exports those receipts through
`GET /receipts/{id}` after rejecting unsafe IDs and missing receipt files.

`GET /models` exposes the same Mac operator catalog as `bitnet mac models`,
including default/supported dense Qwen rows, the BitNet one-shot ask plus fixed
warm-session row, candidate/rejected rows, cache state, disk-headroom guidance,
and the receipt-only BitNet proof bridge commands. It does not run generation,
does not fetch artifacts, and does not expose BitNet as a server completion
model. Cache-path fields are redacted in the HTTP view.

`M4-SERVE-002` does not run generation. Readiness reports whether startup
verified the supported model cache, tokenizer authority, `apple-m4-cpu-neon`
backend route, no-hidden-fallback policy, disk/cache state, and receipt
directory. The HTTP readiness view redacts local cache/model/receipt paths.
Missing or invalid cache still prevents startup.

## Receipt Requirements

Every completed generation request should be able to export a receipt with:

- `artifact_kind = "bitnet_apple_m4_local_server_completion"`;
- `server.host`, `server.port`, `server.endpoint`, and `server.request_id`;
- model ID, source, size, SHA256, and SHA256 source;
- tokenizer authority and prompt template;
- requested backend, selected backend, runtime API, and fallback status;
- generated text and generated token IDs;
- prompt token IDs when enabled by receipt policy;
- time to first token, decode timing, total request timing, and memory;
- streaming enabled/disabled status;
- cache verification status;
- optional `trace_id` plus an `observability` block that links server logs,
  response metadata, and per-request receipts under the M4-OBS-001 redaction
  policy;
- claim-boundary fields stating that dense SLM server success does not prove
  BitNet, QK256, Neural Engine, MPSGraph, full Metal inference, or broad M4
  performance.

HTTP receipt export is a redacted view for operator inspection. The local
receipt file remains the canonical artifact for full local evidence.

Failed startup and failed request receipts should record the failing gate when a
receipt directory is available.

## Claim Boundary

This contract may claim only that the M4 local server command/config surface,
initial health/readiness endpoint slice, model catalog endpoint, first local
completion endpoint, and receipt export endpoint are defined.

It must not claim:

- full OpenAI compatibility;
- production-grade concurrency, uptime, or deployment readiness;
- BitNet local-answer quality;
- full `apple-m4-metal` inference;
- Neural Engine execution;
- MPSGraph model inference;
- QK256 support;
- broad M4 performance.
