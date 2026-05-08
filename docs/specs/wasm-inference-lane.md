# WASM Inference Lane Plan

**Status:** planning contract  
**Audience:** `bitnet-wasm`, inference runtime, tokenizer, model-loader, receipt, and browser-demo contributors  
**Claim boundary:** WASM CPU inference is feasible in theory, but the current repository path is scaffolded and must not be represented as real inference until receipt-backed acceptance gates pass.

## Executive summary

BitNet-rs can support WebAssembly inference as a separate receipt-backed backend lane. The lane should cover two deployment shapes:

1. **Browser/client CPU inference:** the browser downloads the WASM runtime and a separate model artifact, keeps model bytes local, runs decode inside a worker, and streams tokens back to the UI.
2. **WASM sandbox inference:** a native, server, Node, or WASI host loads a WASM module and runs parsing/inference behind a sandbox boundary with explicit CPU, memory, and time limits.

The current project already has a useful WASM crate shape, optional inference feature plumbing, async/browser dependencies, and memory/progressive-loading scaffolding. However, current generation and loading paths are not real inference proofs yet: the exported generation API returns an explicit not-ready error when inference is enabled, the richer wrapper returns placeholder output, and the byte loader still converts JS bytes to a Rust `Vec<u8>` before constructing a `virtual://model...` placeholder path.

The lane therefore starts with proof contracts and compile honesty before attempting browser demos or official BitNet model runs.

## Non-negotiable distinction

```text
WASM inference is feasible.
Current bitnet-rs WASM inference is scaffolded.
```

A WASM detection result, successful module initialization, placeholder string generation, model-loader probe, or browser UI demo is not an inference claim.

## Current foundations in the repository

| Foundation | Current state | Planning implication |
|---|---|---|
| `bitnet-wasm` crate | Exists as WebAssembly bindings for BitNet inference. | Keep the lane in a dedicated crate/backend identity instead of overloading native CPU. |
| Browser dependencies | Uses `wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`, `web-sys`, `serde-wasm-bindgen`, and WASM-friendly `getrandom`. | JS/browser integration foundations are present. |
| Optional inference feature | `bitnet-wasm` has an opt-in `inference` feature forwarding to `bitnet-inference/cpu` and `bitnet-inference/rt-wasm`. | Inference should remain opt-in until the runtime compiles and runs honestly. |
| WASM runtime split | `bitnet-inference` exposes `rt-wasm` dependencies. | Keep runtime assumptions explicit and avoid native runtime leakage. |
| Memory/progressive scaffolding | Memory manager, chunked loading, and buffer utilities exist. | Reuse them, but require real high-water marks and source-buffer lifecycle receipts before claiming model residency. |
| Generation API | `generate()` reports that inference is not ready when the feature is enabled. | This is good claim hygiene; preserve explicit failures until real implementation exists. |
| Wrapper generation | `WasmInference::generate` currently returns formatted placeholder text and simulated stats. | Placeholder output must be removed or marked impossible to confuse with inference before demos. |
| Model byte path | `WasmBitNetModel::load_from_bytes` copies `Uint8Array` to `Vec<u8>` and creates a virtual file path placeholder. | Replace with byte-backed GGUF/tokenizer loading before inference claims. |

## Backend identities

Do not label this lane as generic `cpu` or generic `wasm`. Use explicit backend identities in runtime selection, receipts, docs, and UI strings.

| Backend label | Meaning | First allowed proof |
|---|---|---|
| `wasm-cpu-scalar` | Single-threaded WASM CPU kernels without SIMD-specific proof. | Tiny fixture inference or packed-kernel parity with scalar receipt. |
| `wasm-cpu-simd` | WASM SIMD kernels selected and used. | Scalar/SIMD parity plus kernel invocation count greater than zero. |
| `wasm-cpu-threads` | WASM threads/shared memory selected and used. | Threaded worker proof with deployment headers and no scalar-only fallback. |
| `wasm-browser-worker` | Browser worker execution lane. | Worker off-main-thread generation API with abort and memory reporting. |
| `wasm-wasi-sandbox` | WASI or host-embedded sandbox lane. | Host limits plus sandbox receipt for model parsing and decode. |

Strict mode must fail instead of silently downgrading. Non-strict mode may downgrade only if the receipt records the requested backend, selected backend, fallback status, and fallback reason.

Example strict success receipt fragment:

```json
{
  "requested_backend": "wasm-cpu-simd",
  "selected_backend": "wasm-cpu-simd",
  "runtime_api": "wasm",
  "host": "browser",
  "fallback_used": false
}
```

Example non-strict downgrade receipt fragment:

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

## Claim gates

Use the existing hardware proof-stage language for WASM as well: detection is not execution, execution is not parity, parity is not full inference, and full inference is not performance.

| Gate | Required evidence | Claims allowed | Claims forbidden |
|---|---|---|---|
| `wasm_compile_smoke` | `bitnet-wasm` compiles for `wasm32-unknown-unknown` with intended features. | The WASM crate compiles. | Model loading, inference, performance. |
| `wasm_runtime_detected` | Browser/Node/WASI host reports relevant capabilities such as SIMD, threads, memory limits, and worker availability. | Host/runtime capability is detected. | Backend executed inference. |
| `wasm_kernel_smoke` | Tiny WASM kernel executes. | Kernel execution path works. | Full model inference or BitNet correctness. |
| `wasm_kernel_parity` | WASM kernel output matches native scalar reference within a stated tolerance. | That kernel/subgraph matches reference. | Full decode parity. |
| `wasm_tiny_inference` | Byte-backed tiny fixture model/tokenizer loads and emits deterministic token(s). | WASM inference path exists for the fixture. | Official model support, browser-scale performance. |
| `wasm_browser_short_decode` | Worker, separate model artifact, strict hash, short greedy decode, streaming, abort, and memory receipt. | Local browser demo is real for that model/config. | General browser performance or full 2B feasibility. |
| `wasm_bitnet_reference_parity` | Canonical BitNet artifact or validated fixture, packed weights, selected backend, kernel counts, and native CPU reference parity. | BitNet WASM inference is real for that artifact/config. | Speed claims unless benchmark-backed. |
| `wasm_benchmark_backed` | Receipt-backed benchmark with machine/browser context, timing, memory high-water mark, backend, fallback, and kernel stats. | Performance under stated conditions. | Portable speed claims outside the measured context. |

## Receipt requirements

Every WASM proof receipt must include at least:

```json
{
  "claim": "wasm_tiny_inference",
  "runtime_api": "wasm",
  "host": "browser|node|wasi",
  "requested_backend": "wasm-cpu-scalar",
  "selected_backend": "wasm-cpu-scalar",
  "fallback_used": false,
  "strict_mode": true,
  "model": {
    "source": "downloaded_separately|local_file|opfs|indexeddb|fixture",
    "format": "gguf",
    "sha256": "...",
    "bytes": 0,
    "loader_mode": "strict"
  },
  "tokenizer": {
    "source": "gguf|tokenizer_json|explicit|fixture",
    "sha256": "...",
    "fallback_used": false
  },
  "memory": {
    "limit_bytes": 0,
    "high_water_bytes": 0,
    "source_buffers_released": false,
    "kv_cache_bytes": 0
  },
  "kernels": {
    "bitnet_linear_invocations": 0,
    "simd_invocations": 0,
    "scalar_invocations": 0,
    "fallback_invocations": 0
  },
  "timing": {
    "load_ms": 0,
    "prefill_ms": 0,
    "decode_ms": 0,
    "tokens_generated": 0
  }
}
```

Receipts must not contain success-shaped placeholder text for generation. If a path is not implemented, return an explicit unsupported/not-implemented error and emit no inference receipt.

## Model and tokenizer artifact strategy

Do not embed production model weights in the WASM binary. The browser lane must use:

```text
WASM runtime bundle
+ separate model artifact
+ separate tokenizer artifact when not embedded in GGUF
```

Supported source modes should be added in this order:

1. User-supplied local file bytes.
2. Fixture bytes committed only for tiny tests.
3. Fetched model URL with explicit SHA256 and size validation.
4. Cached artifact in IndexedDB or OPFS.
5. Host/WASI file handle or blob abstraction for sandboxed hosts.

Inference must not depend on live Hugging Face, CDN, or other network access after artifacts are loaded and validated.

## Memory plan

The first real implementation must budget for all relevant residency, not just source bytes:

```text
JS ArrayBuffer or Blob source
+ Rust/WASM linear-memory copy, if unavoidable
+ parsed tensor metadata
+ tensor buffers or mapped views
+ packed BitNet weights
+ tokenizer state
+ KV cache
+ decode workspace and activations
```

Implementation priorities:

1. Validate headers and metadata before copying the full model repeatedly.
2. Parse tensor metadata before allocating packed weights.
3. Pack weights once at load time and reuse them during decode.
4. Drop or release source buffers as soon as safe.
5. Cap context aggressively in early browser proofs.
6. Reuse decode workspace and avoid per-token allocation/repacking.
7. Stream tokens from a worker rather than buffering full output.
8. Record memory limit, high-water mark, model bytes, KV-cache bytes, and whether source buffers were released.

The current memory manager/progressive loader is useful scaffolding, but it does not by itself prove real model residency or decode memory behavior.

## Runtime and dependency split

The WASM lane should keep native runtime assumptions out of browser builds:

- No native `tokio` runtime or `mio` assumptions in `wasm32-unknown-unknown` browser paths.
- No blocking thread-pool assumptions unless behind an explicit `wasm-cpu-threads` feature.
- No filesystem path assumptions in core browser loading.
- No model download/network assumptions inside core inference.
- No native SIMD assumptions in `wasm-cpu-scalar`.
- WASM SIMD must be a selected and measured backend, not an implied property of CPU inference.

## PR ladder

### WASM-001 — Proof contract and lane docs

**Goal:** document backend labels, claim gates, receipts, artifact policy, memory plan, and PR sequence.

**Acceptance:**

- WASM detection is not documented as inference.
- Placeholder generation cannot count as inference.
- Browser demos cannot claim BitNet inference without strict receipts.
- Models are documented as separate artifacts.
- Receipts require model hash, tokenizer source, fallback status, memory, timing, and kernel stats.

### WASM-002 — Honest `bitnet-wasm --features inference` compile

**Goal:** make this command compile without native runtime leakage:

```bash
cargo check --target wasm32-unknown-unknown \
  -p bitnet-wasm \
  --no-default-features \
  --features browser,inference
```

**Acceptance:**

- No native tokio/mio runtime assumptions.
- No uncompiled module-gating issues such as missing `crate::memory` or `crate::utils` under the selected features.
- No placeholder generation is exported as successful inference.
- Unsupported inference paths return explicit errors.

### WASM-003 — Byte-backed GGUF and tokenizer loading

**Goal:** replace virtual file placeholders with real byte-backed loading.

**Acceptance:**

```rust
load_model_from_bytes(&[u8])
load_tokenizer_from_bytes(&[u8])
```

- Browser loading makes no filesystem assumptions.
- Loader validates model format, size, metadata, and hash.
- Tokenizer fallback is receipt-visible and fails in strict mode.

### WASM-004 — Worker-safe API

**Goal:** expose a JS-facing API that is safe to run off the main thread.

```ts
loadModel(modelBytes, tokenizerBytes, config)
generate(prompt, generationConfig)
generateStream(prompt, generationConfig)
unload()
getMemoryStats()
abort(requestId)
```

**Acceptance:**

- Decode is intended for worker execution.
- Generation supports abort/cancel.
- Memory stats are queryable.
- Placeholder generation is impossible to confuse with inference.

### WASM-005 — Tiny fixture inference proof

**Goal:** prove one deterministic tiny fixture token before any full model claims.

**Acceptance receipt fragment:**

```json
{
  "claim": "wasm_tiny_inference",
  "runtime_api": "wasm",
  "selected_backend": "wasm-cpu-scalar",
  "fallback_used": false,
  "model_fixture": "tiny",
  "generated_tokens": 1
}
```

### WASM-006 — WASM SIMD packed-kernel smoke

**Goal:** add a real `wasm-cpu-simd` packed BitNet linear primitive.

**Acceptance:**

- Scalar and SIMD outputs match.
- Kernel invocation count is greater than zero.
- Fallback count is zero in strict SIMD mode.
- Receipt records selected backend as `wasm-cpu-simd` only when SIMD actually ran.

### WASM-007 — Browser short decode proof

**Goal:** build a browser worker proof with separate artifact loading and strict receipts.

**Acceptance:**

- Worker performs short greedy decode off the main thread.
- Model artifact is separate from the WASM bundle.
- Model hash and tokenizer source are verified.
- Output streams token events.
- Memory high-water mark and fallback status are recorded.
- No placeholder response is possible.

### WASM-008 — Official BitNet feasibility proof

**Goal:** prove a canonical BitNet artifact/config only after the smaller lane works.

**Acceptance:**

- Official or canonical BitNet GGUF artifact is validated.
- Real tokenizer is used.
- Packed weights are used.
- One greedy token matches native CPU reference.
- `BitNet` linear invocation count is greater than zero.
- `fallback_used=false` in strict mode.
- No speed claim is made without a benchmark receipt.

## Initial task backlog

- Add receipt schema examples for WASM backend labels.
- Add feature-gating audit for `bitnet-wasm` modules that depend on `crate::memory`, `crate::utils`, and `bitnet-inference`.
- Add compile-only CI lane for `wasm32-unknown-unknown` browser/no-inference build.
- Add compile-only CI lane for opt-in inference once WASM-002 is complete.
- Design byte-backed GGUF loader traits in the model layer.
- Design byte-backed tokenizer loading that can fail strict mode instead of using hidden pretrained/network fallback.
- Add placeholder-output guard tests for JS-facing generation APIs.
- Add memory accounting hooks for high-water mark, source-buffer release, KV cache, and packed weights.
- Add tiny fixture criteria and native reference output generation.

## Things not to do first

- Do not start with the full official 2B-class model in the browser.
- Do not embed model weights in the WASM bundle.
- Do not let `generate()` return plausible fake output.
- Do not claim sandboxed inference from a parser or loader probe.
- Do not claim SIMD if scalar fallback ran.
- Do not depend on network access inside inference.
- Do not require browser threads for the first proof.
- Do not benchmark before correctness, byte-backed loading, and receipt coverage are in place.

## Direct path

```text
docs/proof contract
→ wasm compile cleanup
→ byte-backed model/tokenizer loader
→ worker API
→ tiny fixture proof
→ wasm scalar/SIMD kernel parity
→ browser short decode proof
→ official BitNet proof
→ benchmark-backed performance claims
```
