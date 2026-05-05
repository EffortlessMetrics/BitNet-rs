# CPU Path Plan

**Status:** living implementation plan
**Audience:** contributors working on real-model CPU inference, GGUF loading, tokenizer resolution, QK256/I2_S kernels, strict fallback policy, and benchmark receipts
**Last updated:** 2026-05-05

## Executive summary

The CPU path is not blocked by one missing function. It is made of three partially connected systems that need to become one coherent inference lane:

1. **Model and tokenizer authority** — GGUF loading, layout selection, tensor-name normalization, and tokenizer discovery must resolve through deterministic policy. Compatibility-era or minimal-loader fallbacks must not be allowed to masquerade as strict real-model inference.
2. **Packed quantized kernel authority** — QK256/I2_S packed layout, scalar reference kernels, SIMD kernels, runtime dispatch, FFI boundaries, and receipts must describe the same row/block geometry and the same requested-versus-selected kernel.
3. **Real transformer execution** — packed matmul is necessary but insufficient. The CPU lane also needs RMSNorm, RoPE, attention score/value paths, KV-cache append/read, embedding lookup, output head, batching, and prefill/decode scheduling that are visible in tests and receipts.

The shortest useful path is:

1. make loader + tokenizer + packed layout authoritative and strict;
2. make scalar packed reference kernels correct, easy to validate, and receipt-backed;
3. land decode-first AVX2 QK256/I2_S GEMV that is selected only after CPU feature probing;
4. then widen to AVX-512 and NEON only after the AVX2 decode lane is proven.

## Current code surfaces

The relevant code already exists in the right neighborhoods, but must be treated as one lane instead of independent subsystems.

| Concern | Current surfaces | CPU-lane implication |
|---|---|---|
| GGUF loading | `crates/bitnet-models/src/formats/gguf/loader.rs`, `crates/bitnet-models/src/gguf_simple.rs`, `crates/bitnet-models/src/qk256_utils.rs` | Establish one canonical real-inference GGUF entry point and make strict mode reject unsupported or partial model paths. |
| Tokenizer authority | `crates/bitnet-tokenizers/src/auto.rs`, `crates/bitnet-tokenizers/src/gguf_loader.rs`, `crates/bitnet-tokenizers/src/gguf_tokenizer.rs`, `crates/bitnet-tokenizers/src/universal.rs`, `crates/bitnet-cli/src/tokenizer_discovery.rs` | Preserve deterministic resolution and surface tokenizer source in receipts. Strict mode must fail rather than guess. |
| Packed QK256 layout | `crates/bitnet-qk256-layout-core/src/lib.rs`, `crates/bitnet-quantization/src/qk256_dispatch.rs`, `crates/bitnet-models/src/qk256_utils.rs` | Make `bitnet-qk256-layout-core` the one place for block geometry, alignment, row stride, input shape parsing, and validation errors. |
| Packed kernels | `crates/bitnet-quantization/src/i2s_qk256.rs`, `crates/bitnet-quantization/src/i2s_qk256_avx2.rs`, `crates/bitnet-kernels/src/matmul_dispatch.rs`, `crates/bitnet-kernels/src/dispatch_planner.rs`, `crates/bitnet-kernels/src/dispatch_table.rs` | Keep scalar truth and SIMD fast paths comparable through the same packed matrix contract. Dispatch must identify workload shape: decode GEMV versus prefill GEMM. |
| CPU transformer ops | `crates/bitnet-kernels/src/cpu/`, `crates/bitnet-inference/src/backends.rs`, `crates/bitnet-inference/src/layers/`, `crates/bitnet-inference/src/kv_cache_manager.rs` | Build a visible CPU op matrix for norm, RoPE, attention, KV cache, embeddings, output head, and scheduling. |
| Receipts and validation | `crates/bitnet-receipts/`, `crates/bitnet-inference/src/receipts.rs`, `xtask/tests/verify_receipt*.rs`, `crates/bitnet-receipts/tests/backend_contract_e2e.rs` | Record requested backend/kernel, selected backend/kernel, fallback status, CPU features, tokenizer source, workload shape, metrics, and parity data. |
| Benchmarks/profiling | `crates/bitnet-quantization/benches/qk256_gemv.rs`, `crates/bitnet-kernels/benches/kernel_benchmarks.rs`, `benches/kernel_ops.rs`, `scripts/phase2_flamegraph.sh`, `docs/benchmarks/qk256-dequant-benchmark.md` | Standardize decode-first benchmarks and machine-readable receipt artifacts before broad performance claims. |

## Canonical CPU dispatch path

The intended end-to-end CPU lane is:

```text
CLI / server request
  -> backend selection
  -> inference backend selection
  -> canonical GGUF model loading
  -> deterministic tokenizer resolution
  -> canonical QK256/I2_S layout validation
  -> prefill or decode workload classification
  -> CPU kernel dispatch by workload + ISA
  -> scalar / AVX2 / optional AVX-512 / NEON kernels
  -> CPU transformer ops
  -> receipt + benchmark artifact
```

Every handoff should carry enough metadata to make hidden fallback impossible in strict mode.

## Authority contracts

### GGUF/model loading

The CPU path should be organized around mmap-friendly GGUF parsing and immutable packed tensor views:

- parse GGUF metadata once;
- normalize model-family metadata once, including RoPE, GQA/head geometry, vocab size, and tensor-name mapping;
- decide the packed weight layout once;
- expose read-only packed tensor views to kernels;
- avoid repacking or whole-matrix dequantization in steady-state inference;
- fail early when a tensor claims a QK256/I2_S layout but does not match block geometry.

Strict mode must reject minimal, partial, or compatibility loader paths that cannot prove real-model execution.

### Tokenizer resolution

Tokenizer resolution order is:

1. explicit tokenizer override;
2. tokenizer embedded in GGUF metadata, when supported;
3. sibling tokenizer assets next to the model, such as `tokenizer.json` or `tokenizer.model`;
4. failure in strict mode.

Compatibility helpers may exist for diagnostics, but strict inference must not silently fall back to a generic or hardcoded tokenizer.

### Packed QK256/I2_S layout

`bitnet-qk256-layout-core` should become the single layout authority. It should own:

- `QK256_BLOCK = 256` and packed byte geometry;
- row/column/block iteration rules;
- row stride and alignment validation;
- input shape parsing for `[batch, seq, cols]` and `[batch, cols]` workloads;
- the packed matrix/view trait consumed by scalar and SIMD kernels.

A future canonical API can look like:

```rust
pub struct Qk256BlockView<'a> {
    pub packed_codes: &'a [u8],
    pub logical_cols: usize,
}

pub trait PackedWeightMatrix {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn row_stride_bytes(&self) -> usize;
    fn row_blocks(&self, row: usize) -> &[Qk256BlockView<'_>];
}
```

The exact shape can evolve, but all model-loading and kernel code should converge on one shared contract rather than duplicate layout math.

## Kernel matrix

| Kernel/op | Why it matters | First implementation | Fast implementation |
|---|---|---|---|
| Packed QK256/I2_S GEMV | Decode, one token at a time | scalar reference | AVX2 first, NEON next, optional AVX-512 |
| Packed QK256/I2_S GEMM | Prefill | scalar reference | AVX2/NEON tiled kernels, optional AVX-512 |
| Block unpack/dequant | parity, debug, fixture generation | scalar | optional SIMD helper for validation only |
| RMSNorm | every transformer layer | scalar | AVX2 + NEON |
| RoPE | every attention step | scalar | AVX2 + NEON |
| Q·Kᵀ scores | attention | scalar | AVX2 + NEON where profitable |
| Softmax + scaling + masking | attention | scalar | vectorize after profiling |
| A·V value path | attention | scalar | AVX2 + NEON |
| KV-cache append/read | decode | scalar | cache-aware CPU implementation |
| Embedding gather | input path | scalar | cache-aware CPU implementation |
| Output head/logits | every step | scalar | packed/vectorized if layout supports it |

Decode GEMV is the first performance priority. Prefill GEMM has higher arithmetic intensity and can be optimized after the single-token path is honest and measurable.

## Strict-mode and fallback policy

| Mode | Allowed | Not allowed |
|---|---|---|
| `auto` | scalar fallback, tokenizer discovery fallback, reference dequant fallback for diagnostics, explicit fallback reason in receipt | fake success receipts or missing fallback metadata |
| `strict` | only the requested loader/tokenizer/kernel path, with required CPU features and validated layout | minimal-loader fallback, hardcoded tokenizer fallback, full dequantized steady-state inference, missing-op silent CPU reference substitution |

If a user requests `--strict --kernel qk256-avx2-gemv` and runtime selection ends up on scalar, dequantized reference, or an unsupported tokenizer path, the command should fail.

## Parity and tolerance tiers

| Tier | Examples | Acceptance policy |
|---|---|---|
| Bit/pack exact | block pack/unpack, metadata offsets, row stride | exact byte equality |
| Kernel numeric parity | scalar packed vs AVX2/NEON packed | exact integer accumulation when possible; otherwise `rtol=1e-5`, `atol=1e-5` |
| Layer/model parity | layer outputs, logits, greedy tokens | layer outputs `rtol=1e-4`, `atol=1e-4`; logits compare top-1/top-k plus bounded max abs diff; temperature-0 greedy fixtures should match unless divergence is explained |

Use reference dequant and dense matmul only for tests, fuzzing, fixture generation, and diagnostics. Do not use it for steady-state performance claims.

## Receipt schema target

CPU receipts should include enough data to reproduce kernel selection and reject hidden fallback:

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

These fields can be folded into existing receipt schema types, but they must remain queryable by CI and `xtask verify-receipt` style checks.

## Benchmark profiles

Use four profiles and keep their output diffable:

| Profile | Purpose | Required dimensions |
|---|---|---|
| `micro` | single kernel, synthetic blocks, controlled cache state | kernel, ISA, rows, cols, batch, threads |
| `layer` | one transformer block, fixed shapes | layer shape, op timings, selected kernels |
| `prefill` | prompt-only throughput | prompt tokens/sec, prompt length, batch size |
| `decode` | steady-state generation | generated tokens/sec, latency p50/p95, prompt length, generation length |

All benchmark artifacts should report CPU feature set, selected kernel, fallback status, model id/path, tokenizer source, and thread count.

## Standard commands

These are the baseline commands contributors should keep working as the CPU lane evolves:

```bash
cargo test --locked --workspace --no-default-features --features cpu
cargo test --locked -p bitnet-common --no-default-features --features cpu
cargo test --locked -p bitnet-quantization --release --no-default-features --features cpu
cargo bench --locked -p bitnet-quantization --bench qk256_gemv --features cpu
cargo bench --locked -p bitnet-kernels --bench kernel_benchmarks --features cpu
```

The target shape for a receipt-producing CPU decode run is:

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

## PR-sized implementation plan

### CPU-001 — canonical GGUF authority

**Files:** `crates/bitnet-models/src/formats/gguf/loader.rs`, `crates/bitnet-models/src/gguf_simple.rs`, `crates/bitnet-models/src/lib.rs`, and server model-loading callers.

**Goal:** one loader path for real GGUF inference.

**Acceptance:**

- one public real-inference load path;
- load-time quant/layout validation;
- strict mode rejects unsupported or partial GGUFs;
- legacy/simple loaders are either folded in or clearly marked diagnostic-only.

### CPU-002 — tokenizer authority

**Files:** `crates/bitnet-tokenizers/src/auto.rs`, `crates/bitnet-tokenizers/src/gguf_loader.rs`, `crates/bitnet-tokenizers/src/gguf_tokenizer.rs`, `crates/bitnet-tokenizers/src/universal.rs`, `crates/bitnet-cli/src/tokenizer_discovery.rs`.

**Goal:** deterministic tokenizer resolution with no silent strict-mode fallback.

**Acceptance:**

- explicit precedence order is tested;
- resolved source is emitted to receipts;
- strict mode fails when no authoritative tokenizer is available.

### CPU-003 — canonical packed layout crate

**Files:** `crates/bitnet-qk256-layout-core/src/lib.rs`, `crates/bitnet-quantization/src/qk256_dispatch.rs`, `crates/bitnet-models/src/qk256_utils.rs`.

**Goal:** one source of truth for QK256/I2_S geometry and row iteration.

**Acceptance:**

- model loader emits the canonical layout/view;
- scalar and SIMD kernels consume the canonical layout/view;
- duplicate layout math outside the layout crate is removed or delegated.

### CPU-004 — scalar packed reference kernels

**Files:** `crates/bitnet-quantization/src/i2s_qk256.rs`, `crates/bitnet-kernels/src/ffi.rs`, `crates/bitnet-kernels/src/ffi/bridge.rs`.

**Goal:** reference GEMV/GEMM on packed blocks.

**Acceptance:**

- exact pack/layout parity tests;
- golden fixtures for representative row/block shapes;
- debug dequant helpers are test/diagnostic-only.

### CPU-005 — AVX2 decode-first kernel

**Files:** `crates/bitnet-quantization/src/i2s_qk256_avx2.rs`, `crates/bitnet-kernels/src/matmul_dispatch.rs`, `crates/bitnet-kernels/src/dispatch_planner.rs`.

**Goal:** fast packed GEMV for single-token decode on AVX2/FMA hosts.

**Acceptance:**

- selected only when CPUID supports required features;
- parity with scalar reference;
- benchmark shows a meaningful win over scalar for decode-shaped workloads.

### CPU-006 — CPU transformer op lane

**Files:** `crates/bitnet-kernels/src/cpu/`, `crates/bitnet-inference/src/backends.rs`, `crates/bitnet-inference/src/layers/`, `crates/bitnet-inference/src/kv_cache_manager.rs`.

**Goal:** CPU-visible RMSNorm, RoPE, attention score/value, KV-cache, embedding, and output-head ops.

**Acceptance:**

- deterministic layer/block parity fixtures;
- decode loop can execute a real model path without generic hidden fallback;
- decode-critical op timings are available in benchmark artifacts.

### CPU-007 — strict receipts

**Files:** `crates/bitnet-common/src/backend_selection.rs`, `crates/bitnet-inference/src/backends.rs`, `crates/bitnet-receipts/`, `crates/bitnet-cli/`, `xtask/tests/verify_receipt*.rs`.

**Goal:** requested-vs-selected kernel and fallback metadata are always visible and enforceable.

**Acceptance:**

- receipt records requested and selected backend/kernel;
- fallback reason is explicit or null;
- strict mode rejects hidden fallback;
- receipt checks fail when CPU runs an unrequested fallback path.

### CPU-008 — NEON and optional AVX-512 widening

**Files:** new `*_neon.rs` and `*_avx512.rs` modules plus dispatch tests.

**Goal:** widen only after scalar and AVX2 decode are proven.

**Acceptance:**

- NEON parity and speedups on arm64;
- AVX-512 is optional, probed, and never assumed;
- every SIMD lane has scalar parity fixtures.

## Contributor checklist

Before claiming CPU real-model performance:

- [ ] GGUF loader path is canonical for the tested model.
- [ ] Tokenizer source is explicit in logs and receipts.
- [ ] QK256/I2_S layout is validated at load time.
- [ ] Requested kernel and selected kernel match in strict mode.
- [ ] Fallback is either absent or explicitly documented in non-strict receipts.
- [ ] Scalar packed parity passes for the same shapes.
- [ ] Decode benchmark reports generated tokens/sec, latency p50/p95, CPU features, and thread count.
- [ ] Reference dequant/dense path is not used for steady-state performance claims.
