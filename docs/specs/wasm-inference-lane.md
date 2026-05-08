# WASM Inference Lane Plan

## Purpose

This document defines the planning, foundations, claim boundaries, and PR ladder for
making WebAssembly inference a real, receipt-backed BitNet-rs backend lane.

The target statement is intentionally narrow:

> WASM CPU inference is feasible in BitNet-rs, but the current WASM path is
> scaffolded. A WASM claim only becomes valid when model loading, tokenization,
> decode, backend selection, fallback reporting, memory accounting, and kernel
> receipts prove what actually ran.

This lane covers two deployment modes:

1. **Browser/client CPU inference**: a browser downloads the WASM runtime,
   downloads or accepts a separate model artifact, keeps model bytes local, runs
   decode in a worker, and streams tokens back to the UI.
2. **WASM sandbox inference**: a native app, server, Node host, or WASI host runs
   parsing and inference inside a WASM sandbox to reduce blast radius from
   malformed or untrusted model artifacts.

WASM is not a speed claim by itself. It is a runtime boundary and portability
lane. Performance claims require separate fallback-free benchmark receipts.

## Current foundations

The repository already has enough structure to justify a dedicated lane:

- `bitnet-wasm` is a real crate for WebAssembly bindings and is described as
  WebAssembly bindings for BitNet 1-bit LLM inference.
- The WASM crate already depends on `wasm-bindgen`, `wasm-bindgen-futures`,
  `js-sys`, `web-sys`, `serde-wasm-bindgen`, WASM-compatible `getrandom`, and a
  WASM allocator option.
- The default `bitnet-wasm` feature set builds the browser lane without enabling
  inference by default.
- The optional `bitnet-wasm/inference` feature forwards to
  `bitnet-inference/cpu` and `bitnet-inference/rt-wasm`.
- `bitnet-inference` already has an `rt-wasm` feature that can be used to keep
  browser-friendly async/runtime concerns separate from native runtime concerns.
- WASM memory, progressive loading, streaming, and model wrapper modules exist,
  although some are currently gated or depend on disabled memory modules.

These are foundations, not proof of inference.

## Current claim boundary

The present state must be described as scaffolded:

- The default browser build intentionally avoids inference.
- The exported `generate(prompt)` path returns an explicit not-ready error when
  the `inference` feature is enabled.
- The richer `WasmInference::generate` wrapper returns placeholder text and
  placeholder stats instead of calling the real decode engine.
- The model loading path copies JavaScript `Uint8Array` data into Rust memory and
  still creates a `virtual://model...` placeholder path instead of using a real
  byte-backed GGUF/tokenizer loader.
- Runtime/module gating still needs cleanup before
  `cargo check --target wasm32-unknown-unknown -p bitnet-wasm --no-default-features --features browser,inference`
  can be treated as an acceptance gate.

Therefore:

- WASM feature detection is not inference.
- A successful WASM build is not inference.
- Model header parsing is not inference.
- Placeholder generation is not inference.
- A browser demo cannot claim BitNet inference without a strict receipt.
- SIMD support cannot be claimed if scalar fallback ran.
- Sandboxing cannot be claimed without recording host limits and fallback status.

## Backend identities

Do not report this lane as generic `cpu` or generic `wasm`. Receipts and UI
messages should use explicit backend identities:

| Backend identity | Meaning | First valid use |
|---|---|---|
| `wasm-cpu-scalar` | Single-threaded WASM CPU kernels with no SIMD requirement. | Tiny fixture inference proof. |
| `wasm-cpu-simd` | WASM CPU kernels using WebAssembly SIMD. | SIMD parity smoke with strict no-fallback receipt. |
| `wasm-cpu-threads` | WASM CPU kernels using WASM threads/shared memory. | Later browser/host thread proof with deployment headers recorded. |
| `wasm-browser-worker` | Browser worker execution lane. | Worker API proof with abort, streaming, and memory receipt. |
| `wasm-wasi-sandbox` | WASI or host-embedded sandbox lane. | Host-limited sandbox proof with CPU/memory/time limits recorded. |

Example strict success receipt fragment:

```json
{
  "requested_backend": "wasm-cpu-simd",
  "selected_backend": "wasm-cpu-simd",
  "runtime_api": "wasm",
  "host": "browser",
  "fallback_used": false,
  "fallback_reason": null
}
```

Example non-strict fallback receipt fragment:

```json
{
  "requested_backend": "wasm-cpu-simd",
  "selected_backend": "wasm-cpu-scalar",
  "runtime_api": "wasm",
  "host": "browser",
  "fallback_used": true,
  "fallback_reason": "wasm simd unavailable"
}
```

Strict mode must fail instead of silently downgrading.

## Artifact and loading contract

Do not embed real model weights into the WASM binary. The browser and sandbox
lanes should use separate artifacts:

```text
wasm runtime bundle
+ separate model artifact
+ separate tokenizer artifact when tokenizer data is not embedded in the model
```

Supported model sources should be introduced in this order:

1. User-supplied local file bytes.
2. Fetched model URL bytes with explicit SHA256 validation.
3. Cached model artifact in IndexedDB or OPFS.
4. Host-provided WASI file or capability handle for sandbox mode.

The core loader must gain byte-backed entry points rather than relying on normal
filesystem paths:

```rust
load_model_from_bytes(&[u8])
load_tokenizer_from_bytes(&[u8])
```

Receipt fragment:

```json
{
  "model": {
    "source": "downloaded_separately",
    "format": "gguf",
    "sha256": "...",
    "bytes": 0,
    "loader_mode": "strict"
  },
  "tokenizer": {
    "source": "gguf|tokenizer_json|explicit",
    "fallback_used": false
  }
}
```

No inference path may depend on Hugging Face, network access, or browser storage
availability once the model and tokenizer bytes have been provided and validated.

## Memory plan

WASM memory is the hard constraint. Large models can accidentally create several
resident copies:

```text
JavaScript ArrayBuffer
+ Rust Vec
+ parsed tensor buffers
+ packed weights
+ KV cache
+ activations/decode workspace
```

The lane must track and reduce copies deliberately:

- Validate model metadata before copying full payloads repeatedly.
- Parse tensor metadata before materializing tensor payloads.
- Pack weights once at load time and reuse them across decode steps.
- Drop source buffers as soon as it is safe.
- Reuse decode workspaces across tokens.
- Cap context length aggressively for browser proofs.
- Stream tokens and receipts instead of accumulating unbounded output.
- Record memory high-water marks in proof receipts.

The existing WASM memory/progressive-loading scaffolding can support this work,
but it must eventually measure real residency rather than conceptual allocation
plans only.

## Worker and host API direction

The first product-useful browser API should be worker-safe and avoid UI-thread
decode:

```ts
loadModel(modelBytes, tokenizerBytes, config)
generate(prompt, generationConfig)
generateStream(prompt, generationConfig)
unload()
getMemoryStats()
abort(requestId)
```

Acceptance rules:

- Decode must run off the main browser thread for browser demos.
- Generation must support abort/cancel.
- Streaming must not fabricate tokens.
- Memory stats must be available before and after load, prefill, and decode.
- Errors must distinguish unsupported runtime, unsupported feature, OOM, loader
  validation failure, tokenizer failure, and inference-not-implemented.

## Speed and correctness priorities

Correctness and receipt coverage come before speed:

1. Prove a tiny fixture path before attempting official BitNet 2B browser
   inference.
2. Prove one deterministic token before short decode.
3. Add packed-weight load-time conversion before optimizing decode.
4. Add a scalar WASM kernel before SIMD.
5. Add WASM SIMD parity before speed claims.
6. Add threads only after scalar/SIMD browser-worker proofs.
7. Benchmark only after strict fallback-free receipts prove correctness.

For the official BitNet 2B class model, browser inference should be treated as
possible but tight until real memory and token timing receipts prove otherwise.

## Receipt requirements

Every WASM proof receipt must include:

- Requested and selected backend identity.
- Runtime API (`wasm`).
- Host (`browser`, `node`, `wasi`, or another explicit host).
- Strict-mode setting.
- Model source, format, byte length, and SHA256.
- Tokenizer source and fallback status.
- Loader mode and whether mock/minimal fallback tensors were used.
- Prompt token count and generated token count.
- Kernel family, kernel implementation, and kernel invocation counts.
- Fallback status and fallback reason.
- Memory high-water mark.
- Timing for load, tokenize, prefill, first token, decode steady state, and total
  generation when those phases run.

Tiny fixture example:

```json
{
  "claim": "wasm_tiny_inference",
  "runtime_api": "wasm",
  "host": "browser",
  "requested_backend": "wasm-cpu-scalar",
  "selected_backend": "wasm-cpu-scalar",
  "fallback_used": false,
  "model_fixture": "tiny",
  "generated_tokens": 1,
  "kernel": {
    "implementation": "wasm-scalar",
    "invocations": 1
  }
}
```

## PR ladder

### WASM-001 — Proof contract docs

Acceptance:

- Backend identities are documented.
- Claim boundaries are documented.
- Placeholder generation is explicitly disallowed as proof.
- Separate model/tokenizer artifact strategy is documented.
- Required receipt fields are documented.

### WASM-002 — Honest compile gate

Goal:

```bash
cargo check --target wasm32-unknown-unknown \
  -p bitnet-wasm \
  --no-default-features \
  --features browser,inference
```

Acceptance:

- No native tokio runtime assumptions leak into this feature set.
- No uncompiled `crate::memory` or module-gating errors remain.
- Unsupported inference paths return explicit not-implemented errors.
- No placeholder output is exported as successful inference.

### WASM-003 — Byte-backed GGUF/tokenizer loading

Acceptance:

- `load_model_from_bytes(&[u8])` exists for the WASM lane.
- `load_tokenizer_from_bytes(&[u8])` exists where tokenizer bytes are separate.
- Browser loading no longer depends on `virtual://model.gguf` placeholders.
- Loader receipts record hash, byte length, format, tokenizer source, and strict
  mode.

### WASM-004 — Worker-safe WASM API

Acceptance:

- Worker-facing API supports load, generate, stream, abort, unload, and memory
  stats.
- Browser decode runs off the main thread.
- Errors distinguish not-implemented from validation/runtime failures.
- No fake generation is available through the public success path.

### WASM-005 — Tiny fixture inference proof

Acceptance:

- A tiny committed fixture loads from bytes.
- Tokenizer bytes or fixture tokenizer metadata load from bytes.
- One deterministic token or tiny fixture output is generated.
- Receipt reports `selected_backend=wasm-cpu-scalar`, `fallback_used=false`, and
  `generated_tokens=1`.

### WASM-006 — WASM SIMD packed-kernel smoke

Acceptance:

- A real `wasm-cpu-simd` packed linear primitive exists.
- Scalar and SIMD outputs match within the documented tolerance.
- Strict SIMD mode records SIMD kernel invocation count greater than zero.
- Strict SIMD mode records zero scalar fallback invocations.

### WASM-007 — Browser short decode proof

Acceptance:

- Browser worker runs short greedy decode.
- Model is provided as a separate validated artifact.
- Output streams token-by-token.
- Receipt records memory high-water mark, fallback status, kernel invocation
  counts, and timing.

### WASM-008 — Official BitNet WASM feasibility proof

Acceptance:

- Official or canonical BitNet GGUF artifact is used.
- Tokenizer source is strict and real.
- Weights are packed once and reused.
- At least one greedy token matches a native CPU reference path.
- `fallback_used=false`.
- No speed claim is made unless backed by same-model benchmark receipts.

## Non-goals for the first lane

Do not start by:

- Running the full official 2B model in-browser.
- Embedding model weights in the WASM bundle.
- Returning plausible fake text from `generate()`.
- Claiming sandboxed inference from a loader probe.
- Claiming SIMD when scalar fallback ran.
- Depending on network access inside inference.
- Requiring WASM threads for the first proof.
- Publishing speedup claims before correctness and receipt coverage.

## First milestone definition

The first honest milestone is:

```text
bitnet-wasm builds with inference feature for wasm32
tiny fixture model loads from bytes
tokenizer loads from bytes
one deterministic token or tiny fixture output runs
fallback_used=false
receipt emitted
```

That proves a WASM inference path exists. It does not prove browser BitNet 2B is
fast.
