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

## External Reference Architecture

The CPU lane should follow the same architectural lesson as `bitnet.cpp`: CPU inference for ternary BitNet models is not generic dense inference with a different file extension. Mixed-precision matrix multiplication dominates runtime, so the useful path is specialized I2_S/TL-style packed compute, not repeated whole-matrix dequantization into generic BLAS-shaped tensors.

GGUF is the storage contract that makes this possible in Rust. Treat it as a single-file, metadata-rich, mmap-friendly container whose tensor alignment and metadata are validated once at load time, then turned into immutable packed views. The steady-state inference contract is:

- parse metadata once;
- validate model family, tensor names, RoPE/GQA/head metadata, and quant layout once;
- expose read-only packed tensor views;
- dispatch directly to packed scalar/SIMD kernels;
- never repack or fully dequantize weights on the hot path unless the run is explicitly diagnostic.

The tokenizer side needs the same authority. Hugging Face-style model trees commonly carry `tokenizer.json` and `tokenizer_config.json` next to weights, but runtime discovery must not guess indefinitely. Strict inference requires deterministic precedence and an explicit receipt source.

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

### Hardware planning targets

Plan by ISA lane before planning by specific machine. AVX2 is the first x86 fast-path target because it covers the widest practical CPU set; AVX-512 and NEON should widen only after scalar and AVX2 decode paths are receipt-proven.

| Machine lane | Planning target | Assumption for implementation work |
|---|---|---|
| 8250U CPU lane | AVX2 baseline | Low-core-count, memory-sensitive, decode-first, no wider-ISA assumptions. |
| 258V CPU lane | AVX2 baseline | Current x86 reference lane; newer ISA features remain optional until probed and benchmarked. |
| 5700X | AVX2 baseline | Strong multi-core prefill machine, but decode still drives optimization value. |
| 9950X3D | AVX2 baseline plus optional advanced x86 | Prove AVX2 first; enable wider lanes only with CPUID and receipt-backed speedups. |
| M4 Mac Mini | NEON baseline | Prioritize NEON; keep AMX/Accelerate out of the first CPU milestone. |

| Kernel target | Primary workload | First lane | Acceptance |
|---|---|---|---|
| Scalar packed F32 GEMV | decode diagnostic/reference | all CPUs | Precise `qk256-scalar-f32-gemv` identity; useful diagnostic/oracle path, not a substitute for scaled BitNet I8_S. |
| Scalar packed F32 GEMM | prefill diagnostic/reference | all CPUs | Precise `qk256-scalar-f32-gemm` identity; deterministic no-scale scalar prefill reference. |
| Scalar scaled I2_S x I8_S GEMV | decode correctness | all CPUs | Precise `qk256-scalar-i8s-scaled-gemv` identity; production scalar BitNet decode oracle. |
| Scalar scaled I2_S x I8_S GEMM | prefill correctness | all CPUs | Precise `qk256-scalar-i8s-scaled-gemm` identity; production scalar BitNet prefill oracle. |
| AVX2/FMA packed GEMV | decode performance | mainstream x86-64 | Meaningful speedup over scalar; selected only with CPUID support. |
| AVX2/FMA packed GEMM | prefill performance | mainstream x86-64 | Tiled prefill path after decode GEMV is proven. |
| AVX-512 packed GEMV/GEMM | optional x86 widening | AVX-512 hosts only | Optional and benchmark-proven; never the only fast path. |
| NEON packed GEMV | ARM decode performance | arm64 | First serious Apple/Arm CPU lane. |
| NEON packed GEMM | ARM prefill performance | arm64 | Follow NEON decode proof. |
| Reference dequant + dense matmul | tests and diagnostics | all CPUs | Not valid for steady-state performance claims. |

### Scalar CPU Productization Rails

The scalar lane is governed by the CPU scalar specs:

- [CPU scalar kernel contract](../specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md)
- [CPU scalar hot-path contract](../specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md)
- [CPU scalar parity contract](../specs/BITNET-SPEC-CPU-SCALAR-PARITY.md)
- [CPU scalar performance contract](../specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md)

Scalar has two distinct meanings. The F32/no-scale QK256 path is a diagnostic
and dequant-style reference path. The scaled I2_S x I8_S path is the production
scalar BitNet path and must be selected for real BitNet I2_S tensors with inline
scale. Receipts must report precise kernel IDs instead of the compatibility
aliases `qk256-scalar-gemv` and `qk256-scalar-gemm`.

Strict scalar receipts must show `fallback_used=false` and
`requested_kernel == selected_kernel`. Strict accelerated requests must fail
rather than silently selecting scalar. Scalar performance receipts are baseline
evidence only and must not claim speedup.

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

## Optimization Rules

Use these rules to keep performance work aligned with correctness and receipts.

1. **Keep packed weights packed.** Fuse block decode/unpack, scale, and dot-product work inside the kernel; accumulate in the most stable integer or mixed-precision domain available; scale late.
2. **Separate prefill and decode.** Prefill wants tiled GEMM-like blocking; decode wants tight GEMV-like cache behavior. Shared helpers are fine, but a single abstraction must not hide workload differences.
3. **Own the memory layout.** `bitnet-qk256-layout-core` should define block geometry, alignment, row/block iteration, and the executable view emitted by GGUF loading.
4. **Use direct intrinsics where they matter.** Scalar code is the truth path; `std::arch` AVX2/AVX-512/NEON code is the fast path; portable-SIMD can be considered later as a fallback, not as the first performance authority.
5. **Thread the outer dimension conservatively.** Prefill can parallelize rows/output tiles/layers when working sets remain sane; decode may slow down if extra threads increase cache and KV traffic.
6. **Profile real shapes in the right order.** Optimize packed GEMV decode first, then KV-cache access, RMSNorm/RoPE, output head, and finally prefill GEMM.

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

Use five stable profiles and record the same fields every time so CI/manual receipts remain diffable.

| Profile | Purpose |
|---|---|
| `micro` | Single kernel, synthetic blocks, controlled cache state. |
| `layer` | One transformer block with fixed shapes. |
| `prefill` | Prompt-only throughput. |
| `first_token` | First generated-token latency after prompt processing. |
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
cargo run --locked -p bitnet-bench-receipts --bin cpu_benchmark_receipt --no-default-features -- \
  --kernel qk256-avx2-gemv \
  --strict \
  --selected-backend intel-i5-8250u-cpu-avx2 \
  --model-repo microsoft/bitnet-b1.58-2B-4T-gguf \
  --model-file ggml-model-i2_s.gguf \
  --model-sha256 <sha256> \
  --tokenizer-source gguf_metadata \
  --prompt-tokens 512 \
  --generated-tokens 128 \
  --receipt-out ci/receipts/cpu-avx2-benchmark.json
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
    "reference_kernel": "qk256-scalar-i8s-scaled-gemv",
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
| CPU-SCALAR-000 | Scalar specs and tracker rails | `docs/specs/BITNET-SPEC-CPU-SCALAR-*.md`, `plans/cpu-scalar/implementation-plan.md`, `docs/bitnet/**`, `docs/tracking/campaigns/cpu-proof/active.toml` | Docs-only scalar contract defines precise kernel IDs, hot-path counters, parity direction, performance profiles, and claim boundaries. |
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
5. Scalar precise IDs, strict selection metadata, hot-path counters, answer receipts, and phase baselines.
6. AVX2 decode GEMV.
7. CPU transformer decode ops.
8. Strict receipts and fallback enforcement.
9. BitNet phase benchmarks.
10. Wider ISA lanes.

### Suggested milestone timeline

| Milestone | Focus | Exit proof |
|---|---|---|
| Week 1 | GGUF and tokenizer authority | Strict load/tokenizer failures are deterministic and receipt-visible. |
| Week 2 | Strict-mode fallback policy | Hidden fallback becomes a hard failure in strict proof runs. |
| Week 3 | Scalar packed reference kernels | Exact layout fixtures and scalar GEMV/GEMM parity pass. |
| Week 4 | Layer/block parity fixtures | Transformer block fixtures prove decode-critical CPU math. |
| Week 5 | AVX2 decode GEMV | CPUID-gated AVX2 beats scalar and records selected kernel. |
| Week 6 | KV-cache, RMSNorm, and RoPE tuning | Decode traces identify the next real bottleneck. |
| Week 7 | NEON decode path | arm64 scalar parity and receipt-backed speedup exist. |
| Week 8 | AVX2 prefill GEMM | Prefill benchmarks are separate from decode benchmarks. |
| Week 9 | Receipts and CI benchmarks | Machine-readable CPU artifacts are published by stable commands. |
| Later | Optional AVX-512 lane | Wider x86 lane is probed, optional, and benchmark-proven. |

## Review Checklist

- [ ] Does the change preserve a single GGUF authority for strict real-model inference?
- [ ] Does tokenizer resolution have deterministic precedence and a recorded source?
- [ ] Does the code consume canonical packed layout directly, without hot-path repacking?
- [ ] Is scalar packed parity available before SIMD performance is claimed?
- [ ] Is AVX2/AVX-512/NEON selection gated by runtime feature detection?
- [ ] Does strict mode fail rather than silently substituting fallback execution?
- [ ] Does the receipt include requested and selected backend/kernel plus fallback reason?
- [ ] Does the benchmark name its phase: micro, layer, prefill, or decode?

## Actionable PR Checklist

Use this checklist when slicing implementation work:

- [ ] Remove split-brain GGUF loading paths from strict real-model execution.
- [ ] Make tokenizer resolution explicit, deterministic, and receipt-visible.
- [ ] Make `bitnet-qk256-layout-core` the one QK256/I2_S layout authority.
- [ ] Land scalar packed GEMV/GEMM reference kernels before claiming SIMD speedups.
- [ ] Split scalar receipts into precise F32/no-scale and scaled I2_S x I8_S kernel IDs.
- [ ] Route inline-scale BitNet QK256 through selected scaled scalar metadata, not direct untracked calls.
- [ ] Land AVX2 decode GEMV before wider x86 or ARM lanes.
- [ ] Add RMSNorm, RoPE, KV-cache, attention score/value, embedding gather, and output-head helpers for CPU decode.
- [ ] Record requested versus selected backend and kernel in receipts.
- [ ] Fail strict mode on hidden fallback, minimal-loader fallback, tokenizer fallback, or dequantized steady-state substitution.
- [ ] Add micro, layer, prefill, and decode benchmarks with stable output fields.
- [ ] Publish reproducible receipt JSON for CI/manual CPU proof runs.

## Open Questions and Review Limits

This document is intentionally a planning contract, not a claim that every referenced API already exists with these exact signatures. Before implementation, inspect current file bodies for:

- exact existing loader and tokenizer function names;
- whether transformer CPU ops already exist under different module names;
- current CLI spellings for strict mode, kernel selection, and receipt output;
- receipt schema version and whether CPU kernel fields already have equivalent names.

Those details can change the patch shape, but not the required direction: authoritative GGUF/tokenizer loading, canonical packed layout, scalar truth kernels, decode-first AVX2, then wider ISA lanes.

## I2_S lane note (2026-05-19)

The official Microsoft 2B I2_S/QK256 route remains product-CLI-ready but not globally speed-qualified. New CPU work must preserve explicit scaled-I2_S×I8_S kernel identity and fallback-explicit receipts.

See `docs/bitnet/i2s/README.md` and `plans/i2s/implementation-plan.md` for current lane sequencing.

## Related Documents

- `docs/bitnet/BITNET_MODEL_CONTRACT.md`
- `docs/bitnet/BITNET_QUANTIZATION_CONTRACT.md`
- `docs/bitnet/BITNET_KERNEL_MATRIX.md`
- `docs/bitnet/BITNET_RECEIPT_FIELDS.md`
- `docs/bitnet/BITNET_RUNTIME_PHASES.md`
- `docs/bitnet/BITNET_BENCHMARK_PROTOCOL.md`
- `docs/reference/strict-mode-api.md`
- `docs/reference/tokenizer-discovery-api.md`
