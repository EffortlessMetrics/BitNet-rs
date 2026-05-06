# BitNet CPU Path Plan

## Purpose

This document turns the CPU-path investigation into an implementation contract. The goal is to make BitNet-rs CPU inference an honest, deterministic, receipt-backed lane rather than a collection of loader, tokenizer, layout, and kernel paths that can silently disagree.

Core instruction:

```text
Do not make BitNet "run" by routing around BitNet. Real CPU support means real GGUF loading, real tokenizer resolution, canonical packed layout, scalar packed reference correctness, explicit SIMD dispatch, full transformer decode coverage, strict fallback behavior, and receipt-backed benchmarks.
```

The CPU path is considered production-ready only when a strict run can prove all of these properties:

- the GGUF loader path is canonical and rejects unsupported real-model layouts early;
- tokenizer resolution follows a deterministic policy and never uses compatibility-era fallback in strict mode;
- QK256/I2_S packed layout metadata has one authority and is consumed directly by kernels;
- scalar packed kernels provide the correctness oracle;
- AVX2 decode kernels are selected only when runtime CPU feature detection supports them;
- transformer decode-critical ops are present for CPU execution;
- receipts record requested versus selected backend, kernel, tokenizer source, and fallback status.

## Current Diagnosis

The repo already contains the major surfaces required for a serious CPU lane, but they are not yet unified into a single end-to-end execution story.

| Area | Current surface | Build-upon direction |
|---|---|---|
| GGUF loading | `crates/bitnet-models/src/formats/gguf/loader.rs`, `crates/bitnet-models/src/gguf_simple.rs` | Fold real-model inference onto one canonical loader and make any minimal/simple path diagnostic-only. |
| Tokenizer authority | `crates/bitnet-tokenizers/src/gguf_loader.rs`, `crates/bitnet-tokenizers/src/gguf_tokenizer.rs`, `crates/bitnet-tokenizers/src/auto.rs`, `crates/bitnet-tokenizers/src/universal.rs`, `crates/bitnet-cli/src/tokenizer_discovery.rs` | Centralize deterministic tokenizer precedence and expose tokenizer source in receipts. |
| Packed QK256/I2_S kernels | `crates/bitnet-quantization/src/i2s_qk256.rs`, `crates/bitnet-quantization/src/i2s_qk256_avx2.rs`, `crates/bitnet-quantization/src/qk256_dispatch.rs` | Preserve this lane, but route it through canonical packed-layout types and receipt-backed dispatch. |
| Model-side quant/layout helpers | `crates/bitnet-models/src/quant/i2s_qk256.rs`, `crates/bitnet-models/src/quant/i2s_qk256_avx2.rs`, `crates/bitnet-models/src/qk256_utils.rs` | Remove split-brain layout interpretation; loader output should already match executable packed views. |
| Layout crates | `crates/bitnet-qk256-layout-core`, `crates/bitnet-qk256-dispatch` | Make layout-core the block geometry, alignment, and iteration authority. |
| Generic kernel dispatch | `crates/bitnet-kernels/src/matmul_dispatch.rs`, `crates/bitnet-kernels/src/dispatch_planner.rs`, `crates/bitnet-kernels/src/dispatch_table.rs` | Select kernels by workload phase and ISA, not just by generic matmul availability. |
| FFI boundary | `crates/bitnet-kernels/src/ffi.rs`, `crates/bitnet-kernels/src/ffi/bridge.rs` | Keep APIs stable while recording selected kernel IDs and fallback status. |
| Backend control plane | `crates/bitnet-common/src/backend_selection.rs`, `crates/bitnet-inference/src/backends.rs` | Enforce strict-mode selection and surface receipt hooks here. |
| Validation and receipts | `crates/bitnet-receipts/**`, `tests/**`, `docs/bitnet/*` | Extend existing proof culture with CPU-kernel, tokenizer, layout, and decode receipts. |
| Bench/profiling | `benches/kernel_ops.rs`, `crates/bitnet-kernels/benches/kernel_benchmarks.rs`, `crates/bitnet-quantization/benches/qk256_gemv.rs`, `scripts/phase2_flamegraph.sh` | Standardize micro, layer, prefill, and decode profiles with diffable output. |

## Canonical CPU Dispatch Path

The CPU lane should read as one path:

```text
CLI/server request
  -> backend_selection.rs
  -> bitnet-inference/backends.rs
  -> canonical GGUF loader
  -> canonical tokenizer resolver
  -> QK256/I2_S layout validation
  -> prefill/decode workload classification
  -> scalar | AVX2 | AVX-512 | NEON kernel selection
  -> CPU transformer ops
  -> receipts and benchmark artifacts
```

Strict mode must fail if any requested part of this path is replaced by an unrequested fallback. Auto mode may use scalar or diagnostic fallback paths, but receipts must still record that fallback explicitly.

## Strict-Mode Semantics

| Mode | Allowed | Not allowed |
|---|---|---|
| `auto` | Scalar fallback, tokenizer discovery fallback, reference dequant fallback for diagnostics. | Fake success receipts or missing fallback fields. |
| `strict` | Only the requested loader, tokenizer, backend, layout, and kernel path. | Minimal-loader fallback, hardcoded tokenizer fallback, full dequantized steady-state inference, or silent CPU reference substitution. |

Hard rule:

```text
If `--strict --kernel qk256-avx2-gemv` was requested and the runtime selected scalar, dequantized, or diagnostic execution, the run must fail rather than emit a warning-only receipt.
```

## Loader and Tokenizer Authority

### GGUF loader requirements

- Parse GGUF metadata once.
- Normalize model family, tensor names, RoPE parameters, GQA/head layout, tokenizer references, and quantization metadata before inference starts.
- Validate QK256/I2_S block geometry and alignment at load time.
- Expose immutable packed tensor views that kernels can consume directly.
- Avoid hot-path repacking or whole-matrix dequantization during steady-state inference.
- Keep minimal/simple loader behavior out of strict real-model execution.

Suggested canonical API shape:

```rust
pub fn load_gguf_model(path: &Path, opts: &LoadOptions) -> Result<LoadedBitNetModel>;
```

### Tokenizer resolution requirements

Tokenizer resolution must use this deterministic precedence order:

1. explicit tokenizer override from CLI/API options;
2. tokenizer embedded in or referenced by GGUF metadata, when available;
3. sibling tokenizer assets next to the model, such as `tokenizer.json` and `tokenizer_config.json`;
4. failure in strict mode.

Compatibility fallbacks may exist for tooling, but strict inference must not hardcode GPT-2 or another unrelated tokenizer.

Suggested canonical API shape:

```rust
pub fn resolve_tokenizer(model_path: &Path, opts: &TokenizerOptions) -> Result<ResolvedTokenizer>;
```

## Packed Layout Contract

QK256/I2_S needs one block definition, one alignment contract, one row/block iteration contract, and one conversion point from GGUF metadata into executable layout.

Suggested layout authority:

```rust
pub struct Qk256BlockView<'a> {
    // canonical packed bytes, scale metadata, and block geometry
}

pub trait PackedWeightMatrix {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn row_blocks(&self, row: usize) -> &[Qk256BlockView<'_>];
}
```

Acceptance criteria:

- model loading emits the canonical packed matrix representation directly;
- scalar and SIMD kernels consume the same representation;
- no duplicate repacking is required in steady-state inference;
- layout/pack/unpack tests use exact byte equality.

## Kernel Matrix

Prioritize decode-first CPU execution. Prefill is important, but single-token decode is more sensitive to memory traffic, KV-cache behavior, scalar fallback, and layout conversion.

| Kernel target | Primary workload | First lane | Acceptance |
|---|---|---|---|
| Scalar packed GEMV | decode correctness | all CPUs | Deterministic oracle for SIMD parity. |
| Scalar packed GEMM | prefill correctness | all CPUs | Deterministic oracle and CI fallback. |
| AVX2/FMA packed GEMV | decode performance | mainstream x86-64 | Meaningful speedup over scalar; selected only with CPUID support. |
| AVX2/FMA packed GEMM | prefill performance | mainstream x86-64 | Tiled prefill path after decode GEMV is proven. |
| AVX-512 packed GEMV/GEMM | optional x86 widening | AVX-512 hosts only | Optional and benchmark-proven; never the only fast path. |
| NEON packed GEMV | ARM decode performance | arm64 | First serious Apple/Arm CPU lane. |
| NEON packed GEMM | ARM prefill performance | arm64 | Follow NEON decode proof. |
| Reference dequant + dense matmul | tests and diagnostics | all CPUs | Not valid for steady-state performance claims. |

Suggested scalar APIs:

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

Suggested AVX2 decode API:

```rust
pub unsafe fn qk256_gemv_avx2(
    w: &impl PackedWeightMatrix,
    x: &[f32],
    y: &mut [f32],
) -> Result<()>;
```

## CPU Transformer Op Lane

Packed matmul alone is not real transformer execution. The CPU lane also needs deterministic implementations and parity tests for decode-critical operations.

| Operation | Why it matters | First implementation | Fast implementation |
|---|---|---|---|
| RMSNorm | every layer | scalar | AVX2 and NEON |
| RoPE | every attention step | scalar | AVX2 and NEON |
| Q·Kᵀ score step | attention | scalar | AVX2 and NEON |
| softmax, scaling, masking | attention | scalar | vectorized where profitable |
| A·V step | attention | scalar | AVX2 and NEON |
| KV-cache append/read/stride helpers | decode | scalar | cache-aware CPU implementation |
| embedding gather | input path | scalar | cache-aware CPU implementation |
| logits/output head | every token | scalar | packed/vectorized if supported by layout |

Suggested CPU op APIs:

```rust
pub fn rmsnorm_f32_inplace(x: &mut [f32], weight: &[f32], eps: f32);
pub fn apply_rope_inplace(q: &mut [f32], k: &mut [f32], pos: usize, cfg: &RopeCfg);
pub fn kv_append(cache: &mut KvCache, layer: usize, token: usize, k: &[f32], v: &[f32]) -> Result<()>;
```

## Parity Tolerances

| Tier | Examples | Policy |
|---|---|---|
| Bit/pack exact | metadata, tensor offsets, block pack/unpack | exact byte equality |
| Kernel numeric parity | scalar packed versus AVX2/NEON packed | exact integer accumulation where possible; otherwise tight tolerance |
| Model-level parity | logits, greedy tokens, prompt/decode state | top-k/token agreement plus bounded numeric drift |

Do not invent numeric tolerances inside implementation PRs. Use `docs/bitnet/BITNET_PARITY_TOLERANCES.md` as the policy source and update it deliberately when a new tolerance class is proven. Unknown GPU, OpenVINO, SIMD-reduction, and graph-conversion tolerances must remain `TBD` until receipt-backed parity data exists.

Hard rules:

- scalar packed output is the correctness floor for optimized CPU kernels;
- deterministic greedy tests use temperature `0.0`;
- sampling tests require a seed;
- every parity artifact records max absolute error, mean absolute error, token agreement when applicable, selected kernel, and reference path.

## Benchmark Profiles

Use four stable profiles and record the same fields every time so CI/manual receipts remain diffable.

| Profile | Purpose |
|---|---|
| `micro` | Single kernel, synthetic blocks, controlled cache state. |
| `layer` | One transformer block with fixed shapes. |
| `prefill` | Prompt-only throughput. |
| `decode` | Steady-state tokens/sec for single-stream and small-batch generation. |

Required measurement fields:

- wall time, median, and p95;
- effective bandwidth when relevant;
- prompt tokens/sec and generated tokens/sec;
- selected backend, selected kernel, fallback flag, and fallback reason;
- CPU architecture, feature set, and thread count;
- model id, quantization format, prompt length, generation length, and batch size.

Representative commands to preserve and standardize:

```bash
cargo test --locked --workspace --no-default-features --features cpu
cargo test --locked -p bitnet-common --no-default-features --features cpu
cargo test --locked -p bitnet-quantization --release --no-default-features --features cpu
cargo bench --locked -p bitnet-quantization --bench qk256_gemv --features cpu
cargo bench --locked -p bitnet-kernels --bench kernel_benchmarks --features cpu
```

Target receipt-producing command shape:

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

## CPU Receipt Shape

A CPU receipt must make fallback impossible to hide:

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
    "repo": "microsoft/bitnet-b1.58-2B-4T-gguf",
    "file": "ggml-model-i2_s.gguf",
    "sha256": "TBD",
    "family": "bitnet_b1_58",
    "quant_format": "i2_s"
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
    "prompt_tps": null,
    "decode_tps": null,
    "latency_ms_p50": null,
    "latency_ms_p95": null
  },
  "parity": {
    "reference_kernel": "qk256-scalar-gemv",
    "max_abs_error": 0.0,
    "mean_abs_error": 0.0
  }
}
```

## PR-Sized Work Items

| ID | Work item | Primary files | Acceptance |
|---|---|---|---|
| CPU-BITNET-001 | Loader authority | `crates/bitnet-models/src/formats/gguf/loader.rs`, `crates/bitnet-models/src/gguf_simple.rs`, `crates/bitnet-models/src/lib.rs`, `crates/bitnet-cli/**` | One authoritative strict GGUF load path; minimal fallback impossible in strict proof mode; loader receipts say `loader.mode=real_gguf`. |
| CPU-BITNET-002 | Tokenizer authority | `crates/bitnet-tokenizers/src/gguf_loader.rs`, `crates/bitnet-tokenizers/src/gguf_tokenizer.rs`, `crates/bitnet-tokenizers/src/auto.rs`, `crates/bitnet-tokenizers/src/universal.rs`, `crates/bitnet-cli/src/tokenizer_discovery.rs` | Deterministic precedence; strict mode fails rather than guessing; tokenizer source reaches receipts. |
| CPU-BITNET-003 | Canonical packed layout | `crates/bitnet-qk256-layout-core/src/lib.rs`, `crates/bitnet-quantization/src/i2s_qk256.rs`, `crates/bitnet-quantization/src/qk256_dispatch.rs`, `crates/bitnet-models/src/quant/**`, `crates/bitnet-models/src/qk256_utils.rs` | Loader and kernels share one QK256/I2_S layout authority; byte-exact layout fixtures pass. |
| CPU-BITNET-004 | Scalar packed truth kernels | `crates/bitnet-quantization/src/i2s_qk256.rs`, `crates/bitnet-kernels/src/matmul_dispatch.rs`, `crates/bitnet-kernels/src/ffi.rs`, `crates/bitnet-kernels/src/ffi/bridge.rs`, `crates/bitnet-kernels/tests/**` | Scalar packed GEMV/GEMM are deterministic; SIMD kernels can compare against scalar packed output. |
| CPU-BITNET-005 | AVX2 decode-first GEMV | `crates/bitnet-quantization/src/i2s_qk256_avx2.rs`, `crates/bitnet-kernels/src/matmul_dispatch.rs`, `crates/bitnet-kernels/src/dispatch_planner.rs`, `crates/bitnet-kernels/src/dispatch_table.rs`, `crates/bitnet-receipts/**`, `benches/**` | CPUID-gated AVX2 GEMV has scalar parity, records requested/selected kernel, and fails strict mode on fallback. |
| CPU-BITNET-006 | CPU transformer decode ops | `crates/bitnet-kernels/src/cpu/**`, `crates/bitnet-transformer/**`, `crates/bitnet-inference/src/backends.rs`, `crates/bitnet-inference/**`, `tests/**` | One real-model decode step can run with real tensors; missing ops fail explicitly; KV-cache append/read is deterministic. |
| CPU-BITNET-007 | Strict receipts and fallback enforcement | `crates/bitnet-common/src/backend_selection.rs`, `crates/bitnet-inference/src/backends.rs`, `crates/bitnet-receipts/**`, `crates/bitnet-receipts-core/**`, `crates/bitnet-bench-receipts/**`, `crates/bitnet-cli/**` | Strict proof fails on hidden fallback and emits machine-readable loader/tokenizer/kernel/backend receipt fields. |
| CPU-BITNET-008 | BitNet phase benchmarks | `crates/bitnet-kernels/benches/**`, `crates/bitnet-quantization/benches/**`, `crates/bitnet-bench-receipts/**`, `docs/bitnet/**` | Micro, layer, prefill, first-token, decode, and context profiles use real BitNet fields and fallback status. |
| CPU-BITNET-009 | Wider ISA lanes | NEON and AVX-512 kernel files, dispatch tables, receipts, tests | NEON and AVX-512 widen proven scalar/AVX2 architecture only; each selected kernel has parity and receipts. |

## Roadmap Order

1. Loader authority.
2. Tokenizer authority.
3. Canonical packed layout.
4. Scalar packed reference kernels.
5. AVX2 decode GEMV.
6. CPU transformer decode ops.
7. Strict receipts and fallback enforcement.
8. BitNet phase benchmarks.
9. Wider ISA lanes.

## Review Checklist

- [ ] Does the change preserve a single GGUF authority for strict real-model inference?
- [ ] Does tokenizer resolution have deterministic precedence and a recorded source?
- [ ] Does the code consume canonical packed layout directly, without hot-path repacking?
- [ ] Is scalar packed parity available before SIMD performance is claimed?
- [ ] Is AVX2/AVX-512/NEON selection gated by runtime feature detection?
- [ ] Does strict mode fail rather than silently substituting fallback execution?
- [ ] Does the receipt include requested and selected backend/kernel plus fallback reason?
- [ ] Does the benchmark name its phase: micro, layer, prefill, or decode?

## Related Documents

- `docs/bitnet/BITNET_MODEL_CONTRACT.md`
- `docs/bitnet/BITNET_QUANTIZATION_CONTRACT.md`
- `docs/bitnet/BITNET_KERNEL_MATRIX.md`
- `docs/bitnet/BITNET_RECEIPT_FIELDS.md`
- `docs/bitnet/BITNET_RUNTIME_PHASES.md`
- `docs/bitnet/BITNET_BENCHMARK_PROTOCOL.md`
- `docs/reference/strict-mode-api.md`
- `docs/reference/tokenizer-discovery-api.md`
