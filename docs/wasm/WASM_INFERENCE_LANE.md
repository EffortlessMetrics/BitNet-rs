# WASM Inference Lane

## Purpose

WASM inference is a feasible BitNet runtime direction, but the repository must
treat it as a separate receipt-backed backend lane rather than as generic CPU
inference or generic WASM capability. The lane exists to prove, step by step,
that BitNet or small-model inference can run in a browser, Node, WASI, or other
WASM host without hiding placeholder generation, scalar fallback, filesystem
assumptions, or native runtime dependencies.

The claim boundary is intentionally conservative:

- **Feasible:** CPU inference through WASM can be built in principle.
- **Current state:** the `bitnet-wasm` crate is scaffolded; it does not yet prove
  real model-backed generation.
- **Target state:** a WASM backend emits strict receipts that identify the host,
  model artifact, tokenizer authority, selected WASM backend, fallback state,
  memory high-water mark, kernel invocation counts, and timing.

## Existing Foundation

The repository already has the right shape for a WASM lane:

- `crates/bitnet-wasm` is a dedicated WebAssembly crate for BitNet bindings.
- The default `bitnet-wasm` feature set builds for browser use without enabling
  the Rust inference engine.
- The optional `inference` feature forwards to `bitnet-inference/cpu` and
  `bitnet-inference/rt-wasm`.
- `bitnet-inference` exposes an `rt-wasm` feature with WASM async/timer support.
- WASM wrapper modules exist for model loading, generation, streaming,
  progressive loading, kernels, benchmarking, and memory-oriented scaffolding.

Those foundations are useful, but they are not sufficient proof. A successful
WASM build, browser feature probe, loader probe, or placeholder generation must
not be counted as BitNet inference.

## Current Claim Boundary

The current exported async generation API is explicit scaffold. With the
`inference` feature enabled, the top-level `generate()` path returns a "not yet
ready" error instead of claiming inference. Without the feature, it reports that
the crate was built without inference.

The richer wrapper is also scaffolded: `WasmInference::generate` produces a
placeholder string and records placeholder stats rather than invoking the real
engine. The asynchronous path simulates work. These paths are useful for API
shape, UI wiring, and smoke tests, but they cannot support inference, parity, or
performance claims.

The current model-loading wrapper also copies browser bytes into a Rust `Vec<u8>`
and then uses a virtual-path placeholder instead of a real byte-backed GGUF or
tokenizer loader. That is the largest practical blocker for browser inference,
because a browser cannot depend on native filesystem paths and large models are
sensitive to duplicate residency.

## Backend Identities

WASM must use explicit backend names. Do not collapse these into generic `cpu` or
`wasm` labels.

| Backend identity | Meaning | First allowed claim |
|---|---|---|
| `wasm-cpu-scalar` | Single-threaded WASM CPU execution without SIMD-specific proof. | Scalar WASM kernel or tiny decode proof ran with `fallback_used=false`. |
| `wasm-cpu-simd` | WASM SIMD lane selected and exercised. | SIMD kernel invocation count is greater than zero and scalar fallback count is zero in strict SIMD mode. |
| `wasm-cpu-threads` | WASM threads/atomics lane selected. | Threaded execution ran under a host with required isolation and worker setup. |
| `wasm-browser-worker` | Browser worker product lane. | Decode runs off the UI thread and streams or returns real model-backed tokens. |
| `wasm-wasi-sandbox` | WASI or host-embedded sandbox lane. | Inference ran inside a constrained WASM sandbox with explicit host resource limits. |

Strict mode must fail rather than silently downgrading. If a caller requests
`wasm-cpu-simd` and the host cannot provide SIMD, the receipt must either record
a failed strict run or explicitly record fallback to `wasm-cpu-scalar` with
`fallback_used=true` and a reason.

## Receipt Contract

Every real WASM inference proof must include these fields in addition to the
normal BitNet receipt fields:

```json
{
  "runtime_api": "wasm",
  "host": "browser|node|wasi|embedded",
  "requested_backend": "wasm-cpu-simd",
  "selected_backend": "wasm-cpu-simd",
  "fallback_used": false,
  "fallback_reason": null,
  "strict_backend": true,
  "model": {
    "source": "downloaded_separately|user_file|opfs|indexeddb|fixture",
    "format": "gguf",
    "sha256": "...",
    "bytes": 0,
    "loader_mode": "strict",
    "byte_backed_loader": true
  },
  "tokenizer": {
    "source": "gguf|tokenizer_json|explicit|fixture",
    "sha256": "...",
    "fallback_used": false
  },
  "wasm": {
    "simd_available": true,
    "threads_available": false,
    "bulk_memory_available": true,
    "worker_used": true,
    "memory_high_water_bytes": 0,
    "source_bytes_dropped_after_pack": true
  },
  "kernel": {
    "implementation": "wasm-scalar|wasm-simd|wasm-threads",
    "bitnet_linear_invocations": 0,
    "fallback_kernel_invocations": 0
  },
  "generation": {
    "prompt_tokens": 0,
    "generated_tokens": 1,
    "streamed": false,
    "placeholder_output": false
  }
}
```

A receipt cannot support a WASM inference claim if any of the following are true:

- the output came from a hard-coded or formatted placeholder response,
- the model bytes were only probed or parsed without decode,
- the tokenizer source is unknown or silently substituted,
- `fallback_used` is missing,
- the selected backend is generic `cpu` or generic `wasm`,
- kernel invocation counts are missing for a kernel proof,
- memory high-water information is missing for browser product claims,
- speed is claimed before correctness, parity, and receipt coverage are proven.

## Model and Tokenizer Strategy

Do not embed model weights in the WASM binary. The product path must keep these
as separate artifacts:

```text
wasm runtime bundle
+ separate model artifact
+ separate tokenizer artifact when needed
```

The browser lane should support:

- user-supplied local files,
- fetched model URLs,
- cached artifacts in IndexedDB or OPFS,
- explicit SHA256 validation before inference,
- strict metadata validation,
- no network dependency during decode.

The implementation path needs byte-backed loaders such as:

```rust
load_model_from_bytes(&[u8])
load_tokenizer_from_bytes(&[u8])
```

A `virtual://model.gguf` placeholder is not enough. The loader should validate
headers and tensor metadata without repeatedly copying the full model, pack
weights once when required, and drop source buffers when safe.

## Memory Plan

Memory is the hardest browser constraint. A naïve implementation can hold all of
these at once:

```text
JS ArrayBuffer
+ Rust Vec copy
+ parsed tensor buffers
+ packed weights
+ KV cache
+ activations
+ decode workspace
```

The lane must prove an explicit memory plan before making browser product
claims:

1. Fetch or receive the model artifact separately from the WASM bundle.
2. Validate hash and metadata before decode.
3. Parse tensor metadata before materializing tensor buffers.
4. Avoid duplicate full-model copies where possible.
5. Pack weights once at load time, never per token.
6. Drop source buffers after safe packing or mapping.
7. Reuse decode workspace across tokens.
8. Cap context length aggressively for early proofs.
9. Stream tokens instead of buffering long generations.
10. Record memory high-water marks in receipts.

## Speed and Correctness Priorities

The reasonable-speed path is ordered by proof value, not ambition:

1. Prove a tiny fixture first.
2. Prove byte-backed model and tokenizer loading.
3. Prove one deterministic generated token.
4. Prove scalar WASM parity against a native CPU reference.
5. Add WASM SIMD kernels for packed BitNet linear operations.
6. Prove SIMD invocation counts and strict no-fallback behavior.
7. Move browser execution into a worker.
8. Add optional WASM threads only after scalar/SIMD correctness is established.
9. Benchmark only after correctness, backend identity, fallback state, and memory
   receipts are reliable.

## Milestones

### First honest milestone: WASM inference path exists

Acceptance:

- `bitnet-wasm` compiles for `wasm32-unknown-unknown` with the inference feature.
- A tiny fixture model loads from bytes.
- A tokenizer loads from bytes or fixture metadata.
- One deterministic token or fixture output is produced by the runtime.
- `fallback_used=false` is recorded.
- A receipt is emitted.

This milestone does **not** mean browser BitNet 2B is fast or production-ready.

### First product-useful milestone: local browser demo is real

Acceptance:

- Browser worker execution is used.
- Model artifact is downloaded or supplied separately.
- Model cache and strict hash validation are in place.
- Short greedy decode runs from a real model path.
- Output is streamed or returned without placeholder text.
- Memory high-water mark and fallback status are recorded.

### First BitNet-specific milestone: BitNet WASM proof

Acceptance:

- Official or canonical BitNet GGUF artifact, or a validated BitNet fixture, is
  used.
- Tokenizer authority is real and recorded.
- Packed weights are used where applicable.
- `wasm-cpu-simd` or `wasm-cpu-scalar` is selected explicitly.
- BitNet linear invocation count is greater than zero.
- Native CPU reference parity is recorded for the claimed scope.
- `fallback_used=false` is recorded.
- No speed claim is made unless a separate benchmark receipt supports it.

## PR Ladder

| PR | Title | Goal | Acceptance summary |
|---|---|---|---|
| WASM-001 | WASM proof contract docs | Establish the lane, claim boundary, backend labels, receipt fields, and milestone ladder. | Documentation only; placeholder generation cannot count as inference. |
| WASM-002 | Honest inference-feature compile | Make `bitnet-wasm --features browser,inference` compile for `wasm32-unknown-unknown`. | Compile works even if runtime returns explicit unsupported/not-implemented errors. |
| WASM-003 | Byte-backed GGUF/tokenizer loading | Replace virtual-path placeholders with byte-backed loader APIs. | `load_model_from_bytes(&[u8])` and tokenizer byte loading exist without filesystem assumptions. |
| WASM-004 | Worker-safe API | Expose worker-oriented JS bindings. | `loadModel`, `generate`, `generateStream`, `unload`, and `getMemoryStats` are worker-safe and do not fake output. |
| WASM-005 | Tiny fixture inference proof | Produce the first real tiny-fixture token receipt. | `runtime_api=wasm`, `selected_backend=wasm-cpu-scalar`, `fallback_used=false`, `generated_tokens=1`. |
| WASM-006 | WASM SIMD packed-kernel smoke | Add and prove a packed BitNet linear SIMD primitive. | Scalar/SIMD parity and strict SIMD no-fallback receipt. |
| WASM-007 | Browser short decode proof | Prove short greedy decode in a worker with cached model artifacts. | Streamed real output, model hash, memory high-water mark, and fallback status recorded. |
| WASM-008 | Official BitNet feasibility proof | Attempt official/canonical BitNet WASM proof only after smaller proofs land. | One greedy token, real tokenizer, packed weights, native reference parity, no unbenchmarked speed claim. |

## Non-Goals Until Earlier Proofs Land

Do not start by trying to run the full official 2B model in-browser. Do not
embed the model in the WASM bundle. Do not let `generate()` return plausible fake
output. Do not claim sandboxed inference from a loader probe. Do not claim SIMD
if scalar fallback ran. Do not depend on Hugging Face or other network access
inside inference. Do not require browser threads for the first proof. Do not
benchmark before correctness and receipt coverage.
