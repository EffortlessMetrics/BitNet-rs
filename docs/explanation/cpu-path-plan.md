# CPU Path Plan

**Status:** planning guide  
**Audience:** contributors working on real-model CPU inference, QK256/I2_S kernels, GGUF loading, tokenizer resolution, receipts, and CPU benchmarks

## Executive summary

The CPU lane is not blocked by one missing function. It is three partially connected systems that need to become one coherent inference path:

1. **Model and tokenizer authority** — GGUF loading, quantized layout selection, and tokenizer discovery exist in multiple places. Future CPU work should converge on one strict model-loading path and one deterministic tokenizer-resolution policy.
2. **Packed quantized kernel authority** — QK256/I2_S kernel code, dispatch scaffolding, and layout crates already exist, but the runtime should read as one end-to-end packed execution story instead of a collection of interchangeable experiments.
3. **Real transformer execution** — fast packed matmul is necessary but not sufficient. A useful CPU path also needs RMSNorm, RoPE, attention score/value paths, KV-cache append/read helpers, embedding lookup, output head projection, batching, and prefill/decode scheduling.

The intended implementation order is:

1. Make loader, tokenizer, and packed layout authority strict and deterministic.
2. Make scalar packed reference kernels correct and receipt-backed.
3. Make AVX2 decode-first kernels fast enough for real models.
4. Widen later to AVX-512 and NEON once the AVX2/scalar contract is proven.

## Current repo surfaces

The CPU path already has useful building blocks, but they are spread across crates:

| Area | Current repo surface | Planning implication |
| --- | --- | --- |
| Generic CPU kernel experiments | `crates/bitnet-kernels/src/cpu/` | Keep generic helpers, but do not let them hide packed QK256/I2_S requirements. |
| Dispatch plumbing | `crates/bitnet-kernels/src/matmul_dispatch.rs`, `dispatch_planner.rs`, `dispatch_table.rs` | Dispatch is already a first-class concern and should select by workload and ISA. |
| FFI bridge for kernel calls | `crates/bitnet-kernels/src/ffi.rs`, `crates/bitnet-kernels/src/ffi/` | Kernel APIs should remain boundary-stable and receipt-visible. |
| Packed QK256/I2_S CPU code | `crates/bitnet-quantization/src/i2s_qk256.rs`, `i2s_qk256_avx2.rs`, `qk256_dispatch.rs` | Preserve this lane and make it the packed-kernel authority. |
| Model-side quant/layout helpers | `crates/bitnet-models/src/quant/`, `crates/bitnet-models/src/qk256_utils.rs` | Avoid duplicate layout interpretation between model and quantization crates. |
| Layout/dispatch crates | `crates/bitnet-qk256-layout-core/`, `crates/bitnet-qk256-dispatch/` | Use these as the center of gravity for canonical block geometry and iteration. |
| GGUF loading | `crates/bitnet-models/src/formats/gguf/loader.rs`, `crates/bitnet-models/src/gguf_simple.rs` | Remove ambiguity between simple/minimal and real inference loading paths. |
| Tokenizer discovery/loading | `crates/bitnet-tokenizers/src/`, `crates/bitnet-cli/src/tokenizer_discovery.rs` | Consolidate precedence and fail strictly rather than guessing. |
| Backend selection | `crates/bitnet-common/src/backend_selection.rs`, `crates/bitnet-inference/src/backends.rs` | Strict CPU selection can be implemented without inventing a new control plane. |
| Receipts/tests/benchmarks | `crates/bitnet-receipts/`, `tests/`, `benches/`, `docs/benchmarks/` | Extend existing validation and benchmarking culture instead of creating parallel tooling. |

The target control flow is:

```text
CLI / server request
  -> backend selection
  -> inference backend
  -> GGUF loader + tokenizer authority
  -> QK256/I2_S layout selection
  -> prefill/decode kernel dispatch
  -> scalar / AVX2 / AVX-512 / NEON implementation
  -> transformer CPU ops
  -> receipt + benchmark artifacts
```

## Design principles

### Keep packed weights packed

The CPU path should compute directly over packed weight blocks. Do not dequantize whole matrices into transient FP32/FP16 buffers during steady-state inference. Full dequantized reference views are useful for tests, parity, and diagnostics, but they must not become the hot path.

The packed kernel contract should prefer:

- fused unpack/decode, scale, and dot-product work;
- block-local accumulation with scale applied late when possible;
- immutable packed tensor views from the loader;
- no duplicate repacking once inference starts.

### Separate prefill and decode kernels

Prefill and decode have different performance shapes:

- **Prefill** is prompt-length driven and more GEMM-like.
- **Decode** is one token at a time and more GEMV-like, with stronger sensitivity to cache behavior and KV-cache traffic.

Shared helpers are fine, but the dispatch API should keep workload type visible so a decode-first packed GEMV kernel does not get forced through a generic prefill abstraction.

### Own one canonical memory layout

The CPU lane needs one authority for:

- QK256/I2_S block geometry;
- alignment requirements;
- row, column, and block iteration;
- tail handling;
- the conversion from GGUF tensor metadata into executable packed views.

`bitnet-qk256-layout-core` should become the preferred home for these invariants, with model loading and kernels consuming the same layout contract.

### Use direct SIMD intrinsics where they matter

The recommended kernel hierarchy is:

1. Scalar packed reference for truth, portability, and CI.
2. `std::arch` AVX2/FMA for mainstream x86-64 decode and prefill acceleration.
3. Optional AVX-512 when feature-probed and benchmark-proven.
4. NEON for arm64 after scalar and AVX2 contracts are stable.
5. Portable SIMD only as a future convenience layer, not the first fast-path authority.

## Minimum CPU kernel matrix

| Kernel / op | Why it matters | First implementation | Fast implementation |
| --- | --- | --- | --- |
| Packed GEMV | Single-token decode | Scalar packed reference | AVX2, NEON, optional AVX-512 |
| Packed GEMM | Prompt prefill | Scalar packed reference | AVX2, NEON, optional AVX-512 |
| Block unpack/dequant | Parity, debug, fixture validation | Scalar | Optional AVX2/NEON helper |
| RMSNorm | Every layer | Scalar | AVX2 + NEON |
| RoPE | Every attention step | Scalar | AVX2 + NEON |
| Q·K^T score step | Attention | Scalar | AVX2 + NEON where profitable |
| Softmax + scaling + masking | Attention | Scalar | Vectorized where profitable |
| Attention A·V step | Attention | Scalar | AVX2 + NEON |
| KV-cache append/read/stride helpers | Decode | Scalar | Cache-aware CPU implementation |
| Embedding gather | Input path | Scalar | Cache-aware CPU implementation |
| Output head/logits | Every generated token | Scalar | Packed/vectorized if layout supports it |

## Strict-mode contract

Strict mode exists to prevent hidden fallbacks from being reported as success.

| Mode | Allowed | Not allowed |
| --- | --- | --- |
| `auto` | Scalar fallback, tokenizer discovery fallback, reference dequant fallback for diagnostics | Fake success receipts or unreported fallback |
| `strict` | Only the requested loader, tokenizer, backend, and kernel path | Minimal-loader fallback, hardcoded tokenizer fallback, full dequantized steady-state inference, missing-op silent CPU reference substitution |

If a user requests `--strict --kernel qk256-avx2-gemv` and the runtime executes scalar or dequantized reference code instead, the run should fail rather than emit a warning.

Receipts should record both the requested and selected path so this is machine-checkable.

## Tokenizer authority

Tokenizer resolution should be deterministic and receipt-visible. The recommended precedence is:

1. Explicit tokenizer override from the CLI/API.
2. Tokenizer data embedded in, or explicitly referenced by, GGUF metadata when the model format path supports it.
3. Sibling tokenizer assets next to the model, such as `tokenizer.json` and `tokenizer_config.json`.
4. Failure in strict mode.

Hardcoded GPT-2 or other compatibility fallbacks belong in compatibility tooling or non-strict migration helpers, not in strict real-model inference.

## GGUF and model-loading authority

The canonical loader should:

- parse GGUF metadata once;
- normalize model-family metadata, RoPE parameters, head/GQA layout, and tensor names;
- validate quant/layout metadata at load time;
- expose read-only packed tensor views that kernels can consume directly;
- avoid split-brain behavior between minimal/simple and real inference loaders;
- fail early in strict mode when a tensor claims a packed quantization layout but does not match the expected block structure.

Suggested entry point shape:

```rust
pub fn load_gguf_model(path: &Path, opts: &LoadOptions) -> Result<LoadedBitNetModel>;
```

## Packed-layout API direction

A future canonical layout API should make rows, columns, blocks, alignment, and tails explicit:

```rust
pub struct Qk256BlockView<'a> {
    // canonical block metadata and borrowed packed bytes
}

pub trait PackedWeightMatrix {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn row_blocks(&self, row: usize) -> &[Qk256BlockView<'_>];
}
```

Kernels should consume this layout directly. The model loader should emit it directly. No steady-state inference path should need to reinterpret or repack the same tensor in a second crate.

## Scalar and AVX2 kernel API direction

Scalar kernels should be the correctness reference:

```rust
pub fn qk256_gemv_scalar(
    w: &impl PackedWeightMatrix,
    x: &[f32],
    y: &mut [f32],
) -> Result<()>;

pub fn qk256_gemm_scalar(
    w: &impl PackedWeightMatrix,
    x: &[f32],
    batch: usize,
    y: &mut [f32],
) -> Result<()>;
```

AVX2 should start with decode-first GEMV:

```rust
pub unsafe fn qk256_gemv_avx2(
    w: &impl PackedWeightMatrix,
    x: &[f32],
    y: &mut [f32],
) -> Result<()>;
```

Dispatch should select AVX2 only when CPUID confirms support and strict mode should reject execution if the selected kernel does not match the requested strict kernel.

## Transformer CPU op direction

Packed matmul must be paired with decode-critical transformer helpers:

```rust
pub fn rmsnorm_f32_inplace(x: &mut [f32], weight: &[f32], eps: f32);
pub fn apply_rope_inplace(q: &mut [f32], k: &mut [f32], pos: usize, cfg: &RopeCfg);
pub fn kv_append(cache: &mut KvCache, layer: usize, token: usize, k: &[f32], v: &[f32]) -> Result<()>;
```

The acceptance bar is not merely that the functions exist. The decode loop should be able to run a real model path without generic hidden fallback, and layer/block parity should be deterministic.

## Parity tolerances

Use three validation tiers:

| Tier | Examples | Starting tolerance |
| --- | --- | --- |
| Bit/pack exact | Pack/unpack, metadata, tensor offsets | Exact byte equality |
| Kernel numeric parity | Scalar packed vs AVX2/NEON packed | Exact integer accumulation when possible; otherwise `rtol=1e-5`, `atol=1e-5` |
| Model-level parity | Logits, greedy tokens, prompt/decode state | Top-k agreement plus bounded drift |

Recommended starting policy:

- Pack/unpack/layout: exact byte equality.
- QK256 block dequant/reference views: max absolute error `<= 1e-6` on the same math path, `<= 1e-5` when vector reductions reorder floating-point operations.
- Layer outputs: `rtol=1e-4`, `atol=1e-4`.
- Logits: compare top-1/top-k index agreement and max absolute difference.
- Greedy decode with fixed seed and temperature `0` should match on fixture prompts; any divergence is a release blocker until explained.

## Benchmarks and receipts

Benchmark at four levels:

| Profile | Purpose |
| --- | --- |
| `micro` | Single kernel with synthetic blocks and controlled cache state |
| `layer` | One transformer block with fixed shapes |
| `prefill` | Prompt-only throughput |
| `decode` | Steady-state generated tokens/sec, single-stream and small-batch |

Each benchmark or receipt should include:

- wall time, median, and p95;
- prompt tokens/sec and generated tokens/sec;
- selected backend and selected kernel;
- fallback flag and fallback reason;
- CPU architecture, feature set, and thread count;
- model path/id, quant format, tokenizer source;
- prompt length, generation length, and batch size;
- parity reference kernel and drift metrics when available.

Suggested receipt shape:

```json
{
  "schema_version": 1,
  "profile": "decode",
  "requested_backend": "cpu",
  "selected_backend": "cpu",
  "requested_kernel": "qk256-avx2-gemv",
  "selected_kernel": "qk256-avx2-gemv",
  "fallback_used": false,
  "fallback_reason": null,
  "cpu": {
    "arch": "x86_64",
    "features": ["avx2", "fma"],
    "threads": 8
  },
  "model": {
    "path": "models/model.gguf",
    "family": "bitnet",
    "quant_format": "i2_s_qk256"
  },
  "tokenizer": {
    "source": "tokenizer.json",
    "strict": true
  },
  "workload": {
    "prompt_tokens": 512,
    "generated_tokens": 128,
    "batch_size": 1
  },
  "metrics": {
    "prompt_tps": 842.1,
    "decode_tps": 18.4,
    "latency_ms_p50": 54.2,
    "latency_ms_p95": 59.8
  },
  "parity": {
    "reference_kernel": "qk256-scalar-gemv",
    "max_abs_error": 0.0,
    "mean_abs_error": 0.0
  }
}
```

Representative commands to keep standardized:

```bash
cargo test --locked --workspace --no-default-features --features cpu
cargo test --locked -p bitnet-common --no-default-features --features cpu
cargo test --locked -p bitnet-quantization --release --no-default-features --features cpu
cargo bench --locked -p bitnet-quantization --bench qk256_gemv --features cpu
cargo bench --locked -p bitnet-kernels --bench kernel_benchmarks --features cpu
```

A future receipt-producing CLI shape should look like:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- infer \
  --model models/bitnet.gguf \
  --prompt-file prompts/wiki_512.txt \
  --max-new-tokens 128 \
  --backend cpu \
  --kernel qk256-avx2-gemv \
  --strict \
  --receipt-out ci/receipts/cpu-avx2-decode.json
```

## PR-sized roadmap

| ID | Work item | Primary files | Acceptance |
| --- | --- | --- | --- |
| CPU-001 | Unify GGUF authority | `crates/bitnet-models/src/formats/gguf/loader.rs`, `crates/bitnet-models/src/gguf_simple.rs`, `crates/bitnet-models/src/lib.rs` | One canonical load path; quant/layout metadata validated; strict mode rejects unsupported or partial models. |
| CPU-002 | Unify tokenizer authority | `crates/bitnet-tokenizers/src/`, `crates/bitnet-cli/src/tokenizer_discovery.rs` | Deterministic precedence; receipt-visible tokenizer source; strict mode fails instead of guessing. |
| CPU-003 | Canonical packed layout crate | `crates/bitnet-qk256-layout-core/src/lib.rs`, `crates/bitnet-quantization/src/qk256_dispatch.rs`, `crates/bitnet-models/src/qk256_utils.rs` | Loader and kernels share one QK256/I2_S block geometry and iteration contract. |
| CPU-004 | Scalar packed reference kernels | `crates/bitnet-quantization/src/i2s_qk256.rs`, FFI bridge files | Exact pack/layout parity tests; stable golden fixtures; dequant helper remains diagnostic/test-only. |
| CPU-005 | AVX2 decode-first kernel | `crates/bitnet-quantization/src/i2s_qk256_avx2.rs`, dispatch planner files | Meaningful speedup over scalar on AVX2 hosts; parity with scalar; selected only when CPUID supports it. |
| CPU-006 | CPU transformer op lane | `crates/bitnet-kernels/src/cpu/`, `crates/bitnet-transformer/src/lib.rs`, `crates/bitnet-inference/src/backends.rs` | RMSNorm, RoPE, attention, KV-cache helpers, and output path are deterministic and decode-ready. |
| CPU-007 | Strict mode and receipts | `crates/bitnet-common/src/backend_selection.rs`, `crates/bitnet-inference/src/backends.rs`, `crates/bitnet-receipts/`, `crates/bitnet-cli/` | Requested/selected kernel and fallback reason are recorded; hidden fallback fails in strict mode. |
| CPU-008 | NEON and optional AVX-512 widening | New `*_neon.rs`, new `*_avx512.rs`, dispatch tables and tests | NEON parity on arm64; AVX-512 optional and never assumed. |

## Hardware planning lanes

Plan by ISA baseline first, not by specific machine SKU:

| Machine lane | Planning target | Assumption |
| --- | --- | --- |
| Low-core x86 laptop | AVX2 baseline | Memory-sensitive, decode-first, no wider-ISA assumptions. |
| Current mainstream x86 | AVX2 baseline | Use as the primary x86 reference lane. |
| High-core x86 desktop | AVX2 baseline | Stronger prefill testing; decode remains optimization priority. |
| Latest x86 desktop | AVX2 baseline plus optional advanced x86 | Enable wider ISA only if probed and benchmark-proven. |
| Apple/Arm desktop | NEON baseline | Prioritize NEON after scalar/AVX2 contracts; keep platform accelerators out of the first CPU milestone. |

## Actionable checklist

- [ ] Remove split-brain GGUF loading paths.
- [ ] Make tokenizer resolution explicit and strict.
- [ ] Make `bitnet-qk256-layout-core` the layout authority.
- [ ] Land scalar packed GEMV/GEMM reference kernels.
- [ ] Land AVX2 decode GEMV first.
- [ ] Add RMSNorm, RoPE, KV-cache, and attention helpers for CPU.
- [ ] Record requested vs selected kernel in receipts.
- [ ] Fail strict mode on hidden fallback.
- [ ] Add micro, layer, prefill, and decode benchmarks.
- [ ] Publish reproducible receipt JSON in CI or manual benchmark artifacts.

## Open questions

These details should be answered from code while implementing the roadmap:

- exact current function signatures in loader, tokenizer, and dispatch modules;
- whether some transformer CPU ops already exist under names outside the obvious CPU modules;
- exact CLI option names for kernel selection and receipt output;
- final home for QK256/I2_S layout types if crate consolidation changes the workspace shape.

These are implementation details, not strategy blockers. The path remains: authoritative GGUF/tokenizer loading, canonical packed layout, scalar truth kernels, AVX2 decode first, then optional widening.
