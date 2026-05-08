# WASM Inference Lane

**Status:** Planning and proof-contract foundation  
**Last updated:** 2026-05-08  
**Scope:** Browser CPU inference and WASM sandbox inference for BitNet-rs

## Executive summary

WASM CPU inference is feasible for BitNet-rs, but the current repository should be treated as scaffolded rather than production-ready for WASM generation. The lane should move forward as an explicit, receipt-backed backend family instead of being folded into generic `cpu` or generic `wasm` claims.

The distinction this lane preserves is:

> WASM inference is feasible. Current BitNet-rs WASM inference is scaffolded until strict byte-backed loading, real decode, and receipts prove otherwise.

The initial goal is not to claim a fast browser run of an official 2B-class BitNet model. The first honest goal is to compile the WASM inference feature, load a tiny byte-backed fixture without filesystem assumptions, run a deterministic token proof, and emit a receipt that records exactly which backend, loader, tokenizer, fallback, memory, and kernel paths executed.

## Current repository foundation

The repository already has the right shape for a WASM lane:

- `bitnet-wasm` is a dedicated crate for WebAssembly bindings.
- Default browser builds intentionally avoid the inference dependency graph.
- An opt-in `inference` feature forwards to `bitnet-inference/cpu` and `bitnet-inference/rt-wasm`.
- WASM-facing dependencies already include `wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`, `web-sys`, `serde-wasm-bindgen`, and WASM-friendly `getrandom`.
- Browser and PWA examples already exist as integration shells.
- The crate contains model, inference, streaming, progressive-loading, and memory-manager scaffolding behind feature gates.

That foundation is useful, but it is not proof of real WASM inference.

## Current non-goals and claim boundaries

The WASM lane must keep claim hygiene strict:

- WASM feature detection is not inference.
- A browser demo shell is not inference.
- A model download/cache UI is not inference.
- Placeholder text generation is not inference.
- Loading metadata or a `virtual://` path is not byte-backed GGUF inference.
- Scalar fallback is not a successful SIMD run.
- Sandboxed parsing is not sandboxed inference unless decode runs in the sandbox.
- Speed claims require benchmark receipts, not screenshots or manually observed latency.

Until this lane reaches the fixture proof milestone, public language should say that WASM inference is planned/scaffolded and that generated output is not yet backed by the real engine.

## Backend identities

Use explicit backend identities so receipts can describe what actually ran:

| Backend identity | Meaning | First valid proof |
| --- | --- | --- |
| `wasm-cpu-scalar` | Single-threaded WASM CPU decode without SIMD-specific kernels | Tiny fixture token proof with `fallback_used=false` |
| `wasm-cpu-simd` | WASM CPU decode using WASM SIMD packed kernels | Scalar/SIMD parity plus kernel invocation count > 0 |
| `wasm-cpu-threads` | WASM CPU decode using browser or host threads | Thread-enabled proof with isolation headers/host capabilities recorded |
| `wasm-browser-worker` | Browser worker orchestration lane around one of the CPU backends | Worker proof with streamed tokens and abort support |
| `wasm-wasi-sandbox` | WASI or host-embedded sandbox lane | Host-limited sandbox proof with CPU/memory/time caps recorded |

Do not collapse these to plain `cpu`, and do not report `wasm-cpu-simd` when the runtime silently selected scalar code.

## Receipt contract

Every WASM proof receipt should include at least the following fields:

```json
{
  "claim": "wasm_tiny_inference",
  "runtime_api": "wasm",
  "host": "browser|node|wasi",
  "requested_backend": "wasm-cpu-simd",
  "selected_backend": "wasm-cpu-scalar",
  "strict_backend": true,
  "fallback_used": true,
  "fallback_reason": "wasm simd unavailable",
  "model": {
    "source": "downloaded_separately|local_file|fixture|opfs_cache|indexeddb_cache",
    "format": "gguf",
    "sha256": "...",
    "bytes": 0,
    "loader_mode": "strict-bytes"
  },
  "tokenizer": {
    "source": "gguf|tokenizer_json|fixture|explicit",
    "sha256": "...",
    "fallback_used": false
  },
  "generation": {
    "prompt_sha256": "...",
    "max_new_tokens": 1,
    "generated_tokens": 1,
    "decode_mode": "greedy"
  },
  "memory": {
    "model_bytes_input": 0,
    "peak_wasm_heap_bytes": 0,
    "kv_cache_bytes": 0,
    "workspace_bytes": 0
  },
  "kernels": {
    "bitnet_linear_invocations": 0,
    "scalar_invocations": 0,
    "simd_invocations": 0,
    "fallback_invocations": 0
  },
  "timing": {
    "load_ms": 0,
    "prefill_ms": 0,
    "decode_ms": 0
  }
}
```

Strict mode must fail if the requested backend cannot be selected. Non-strict mode may downgrade, but only with `fallback_used=true` and a concrete `fallback_reason`.

## Model and tokenizer artifact strategy

Do not embed model weights in the WASM binary. The deployable shape should be:

```text
wasm runtime bundle
+ separately downloaded or user-provided model artifact
+ separately downloaded or user-provided tokenizer artifact when not embedded in GGUF
+ explicit metadata and hash validation
```

The browser path should support these sources:

- User-selected local file.
- Fetched model URL.
- Cached model artifact in IndexedDB or OPFS.
- Fixture artifact for deterministic tests.

Inference must not depend on Hugging Face, network access, or browser fetch once the model/tokenizer bytes are loaded and validated.

## Byte-backed loading requirement

The critical loader requirement is a real byte-backed API, not a virtual path placeholder:

```rust
load_model_from_bytes(&[u8])
load_tokenizer_from_bytes(&[u8])
```

Acceptable future variants include reader/blob/OPFS-handle forms, but the inference core must not require normal filesystem paths for browser execution. The first milestone should parse a tiny GGUF or fixture format directly from bytes, validate its hash, construct model/tokenizer state, and drop source buffers when no longer needed.

## Memory plan

Browser memory pressure is the hard constraint. Large models can accidentally require all of the following at once:

```text
JS ArrayBuffer copy
+ Rust Vec copy
+ parsed tensor buffers
+ packed weights
+ KV cache
+ activations
+ decode workspace
```

The WASM lane should therefore require a memory receipt and design toward:

1. Validate headers and metadata without repeatedly copying full model bytes.
2. Parse tensor metadata before allocating tensor storage.
3. Pack weights once at load time.
4. Drop source buffers as soon as model residency no longer needs them.
5. Reuse decode workspaces.
6. Cap context aggressively in browser proofs.
7. Stream tokens instead of buffering long responses.
8. Record peak WASM heap, KV cache bytes, packed weight bytes, and workspace bytes.

The current memory/progressive-loading scaffolding can be evolved into this proof system, but conceptual allocation tracking is not enough for the first real claim.

## Execution modes

### Browser/client CPU inference

Target flow:

```text
browser downloads wasm runtime
browser obtains model and tokenizer artifacts separately
artifacts are hash-validated and cached
worker loads bytes into the WASM runtime
decode runs off the UI thread
tokens stream back to the UI
receipt records backend, fallback, memory, and kernels
```

This is the product-facing lane. It should start with a tiny fixture or very small SLM and short contexts before attempting larger BitNet artifacts.

### WASM sandbox inference

Target flow:

```text
native app, server, Node, or WASI host loads wasm module
host passes validated model/tokenizer bytes or handles
inference runs inside the wasm sandbox
host enforces CPU, memory, and wall-time limits
receipt records host, limits, model hash, fallback, and decode proof
```

This lane is useful for blast-radius reduction around model parsing and untrusted artifacts. It does not make inference faster by itself and still needs resource limits to prevent denial-of-service behavior.

## PR ladder

### WASM-001 — Proof contract docs

**Goal:** Land this lane contract and align examples/roadmaps with scaffolded status.

**Acceptance:**

- Backend identities are defined.
- Detection, placeholder generation, and browser shells are excluded from inference claims.
- Receipt fields cover model hash, tokenizer source, fallback status, memory, timing, and kernel stats.
- Model/tokenizer artifacts are explicitly separate from the WASM bundle.

### WASM-002 — Honest inference-feature compile

**Goal:** Make the opt-in inference build compile for the browser target, even if unsupported execution returns a clear not-implemented error.

```bash
cargo check --target wasm32-unknown-unknown \
  -p bitnet-wasm \
  --no-default-features \
  --features browser,inference
```

**Acceptance:**

- No native Tokio runtime assumptions leak into the selected feature set.
- No uncompiled `crate::memory` or module-gating issues remain.
- No placeholder response is exported as successful inference.
- Unsupported paths fail with explicit errors.

### WASM-003 — Byte-backed model/tokenizer loading

**Goal:** Replace virtual-file loading with byte-backed model and tokenizer entry points.

**Acceptance:**

- `load_model_from_bytes(&[u8])` exists for the first supported format/fixture.
- `load_tokenizer_from_bytes(&[u8])` exists or tokenizer-in-GGUF is explicitly proven.
- Browser load path has no required filesystem assumption.
- Receipt records loader mode, input bytes, and SHA256.

### WASM-004 — Worker-safe API

**Goal:** Define a JS-facing API that is safe to use from a Web Worker.

```ts
loadModel(modelBytes, tokenizerBytes, config)
generate(prompt, generationConfig)
generateStream(prompt, generationConfig)
unload()
getMemoryStats()
```

**Acceptance:**

- Decode is not run on the UI thread in browser examples.
- Abort/cancel behavior is supported.
- Memory stats are observable.
- Fake generation is removed or clearly rejected.

### WASM-005 — Tiny fixture inference proof

**Goal:** Run one deterministic generated token or fixture output through the WASM path.

**Acceptance receipt:**

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

**Goal:** Add a real `wasm-cpu-simd` packed linear primitive.

**Acceptance:**

- Scalar and SIMD outputs match within the documented tolerance.
- SIMD kernel invocation count is greater than zero.
- Strict SIMD mode reports fallback count zero or fails.

### WASM-007 — Browser short decode proof

**Goal:** Demonstrate a browser worker with separate model artifact, strict hash validation, short greedy decode, streaming output, and memory receipt.

**Acceptance:**

- Worker path emits non-placeholder tokens.
- Model/tokenizer are loaded separately and validated.
- Memory high-water mark is recorded.
- Fallback status is recorded.

### WASM-008 — Official BitNet feasibility proof

**Goal:** Attempt official or canonical BitNet artifact only after the smaller lane works.

**Acceptance:**

- Strict model loader and real tokenizer are used.
- Packed weights are built once and reused.
- One greedy token matches a native CPU reference for the same prompt/config.
- `wasm-cpu-simd` claims require SIMD invocation count > 0 and fallback count 0.
- No speed claim is made unless a benchmark receipt exists.

## First milestone definition

The first milestone that may be called "WASM inference path exists" is:

```text
bitnet-wasm builds with --features browser,inference for wasm32
tiny model fixture loads from bytes
tokenizer loads from bytes or is proven embedded in the fixture
one deterministic token or tiny fixture output runs
fallback_used=false
receipt emitted
```

This milestone does not claim that browser BitNet 2B is fast or product-ready.

## Product-useful milestone definition

The first milestone that may be called "client-side local inference demo is real" is:

```text
browser worker
separate model/tokenizer artifact acquisition
cached artifact with strict hash validation
short greedy decode
streamed non-placeholder output
memory high-water mark receipt
fallback and kernel status receipt
```

## Do not do yet

- Do not start with full official 2B-class browser inference.
- Do not embed model weights in the WASM bundle.
- Do not let `generate()` return plausible fake output as success.
- Do not claim sandboxed inference from a parser or loader probe.
- Do not claim SIMD if scalar fallback ran.
- Do not depend on network access inside inference.
- Do not require browser threads for the first proof.
- Do not publish speed claims before correctness and receipt coverage.

## Open implementation questions

- Which tiny fixture format should be accepted first: GGUF, a purpose-built deterministic fixture, or both?
- Should byte-backed GGUF loading live in `bitnet-models`, `bitnet-inference`, or a shared loader crate?
- What memory-measurement source is authoritative in browser builds: allocator accounting, `performance.memory` when available, or a combined receipt?
- How should receipts be emitted from worker contexts: returned JSON, callback event, downloadable artifact, or all three?
- Which browser support baseline should be required before enabling `wasm-cpu-threads`?
