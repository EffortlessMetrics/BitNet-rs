# CPU Path Plan

**Status:** planning baseline for native Rust CPU inference  
**Scope:** GGUF/tokenizer authority, QK256/I2_S packed kernels, transformer CPU ops, strict fallback policy, and receipt-backed benchmarks  
**Primary goal:** turn the existing partially-connected CPU pieces into one deterministic, measurable inference lane.

## Executive summary

The CPU path is not blocked by one missing function. It is split across three systems that must agree at runtime:

1. **Model and tokenizer authority** — GGUF loading, tensor layout selection, and tokenizer discovery exist, but they must become one strict path for real-model inference.
2. **Packed quantized kernel authority** — QK256/I2_S code and dispatch scaffolding already exist, but model-side layout helpers, quantization kernels, and dispatch crates must converge on one packed layout contract.
3. **Real transformer execution** — fast packed matmul is necessary but insufficient; decode also needs RMSNorm, RoPE, attention score/value paths, KV-cache append/read helpers, embedding lookup, output head, batching, and prefill/decode scheduling.

The implementation priority is:

1. make loader, tokenizer, and layout authority strict and deterministic;
2. land scalar packed reference kernels with parity fixtures;
3. make AVX2 decode-first packed GEMV fast and receipt-backed;
4. then widen to AVX-512 and NEON only after the AVX2 lane is proven.

## Current repo surfaces to preserve and unify

| Area | Current surfaces | Planning implication |
| --- | --- | --- |
| GGUF loading | `crates/bitnet-models/src/formats/gguf/loader.rs`, `crates/bitnet-models/src/gguf_simple.rs` | Fold real inference into one canonical loader path; keep simple/minimal behavior out of strict mode. |
| Tokenizer resolution | `crates/bitnet-tokenizers/src/gguf_loader.rs`, `gguf_tokenizer.rs`, `auto.rs`, `universal.rs`, `crates/bitnet-cli/src/tokenizer_discovery.rs` | Define deterministic precedence and receipt the selected source. |
| Packed QK256/I2_S kernels | `crates/bitnet-quantization/src/i2s_qk256.rs`, `i2s_qk256_avx2.rs`, `qk256_dispatch.rs` | Treat packed QK256/I2_S as the native CPU lane, not a debug-only format. |
| Model-side quant/layout helpers | `crates/bitnet-models/src/quant/i2s_qk256.rs`, `i2s_qk256_avx2.rs`, `qk256_utils.rs` | Remove duplicate layout interpretations by routing through a canonical layout crate. |
| Layout and dispatch crates | `crates/bitnet-qk256-layout-core`, `crates/bitnet-qk256-dispatch` | Make these the source of truth for block geometry, alignment, and iteration. |
| Generic kernel dispatch | `crates/bitnet-kernels/src/matmul_dispatch.rs`, `dispatch_planner.rs`, `dispatch_table.rs`, `ffi.rs`, `ffi/bridge.rs` | Dispatch must record requested/selected kernels and make hidden fallbacks impossible in strict mode. |
| Backend selection | `crates/bitnet-common/src/backend_selection.rs`, `crates/bitnet-inference/src/backends.rs` | Reuse the existing control plane for strict CPU selection and receipts. |
| Tests, receipts, benches | `crates/bitnet-receipts/tests/backend_contract_e2e.rs`, kernel tests, `benches/kernel_ops.rs`, `crates/bitnet-kernels/benches/kernel_benchmarks.rs`, `crates/bitnet-quantization/benches/qk256_gemv.rs`, `scripts/phase2_flamegraph.sh` | Extend the existing validation culture rather than inventing a separate harness. |

## Canonical CPU dispatch path

The native CPU lane should be organized as follows:

```text
CLI/server request
  -> backend selection
  -> bitnet-inference backend
  -> canonical GGUF loader
  -> deterministic tokenizer resolution
  -> canonical QK256/I2_S layout selection
  -> workload-aware kernel dispatch
     -> prefill kernels
     -> decode kernels
        -> scalar reference / AVX2 / optional AVX-512 / NEON
  -> transformer CPU ops
  -> receipt + benchmark artifacts
```

The important invariant is that the loader, layout code, kernels, backend selection, and receipts must all report the same execution story.

## Strict-mode contract

| Mode | Allowed | Not allowed |
| --- | --- | --- |
| `auto` | scalar fallback, tokenizer discovery fallback, reference dequant fallback for diagnostics, explicit fallback receipts | fake success receipts or unrecorded fallback |
| `strict` | only the requested loader/tokenizer/kernel path | minimal-loader fallback, hardcoded tokenizer fallback, full dequantized steady-state inference, missing-op substitution, scalar fallback when a specific SIMD kernel was requested |

Strict mode must fail if the requested path cannot run. For example, if a user requests `--strict --kernel qk256-avx2-gemv` and dispatch selects scalar or dequantized reference code, the command should return an error rather than a warning.

## Loader and tokenizer policy

### GGUF loading requirements

- Parse GGUF metadata once and normalize model-family metadata before execution.
- Prefer mmap-friendly, read-only tensor views where possible.
- Validate tensor names, dimensions, quantization type, block count, and alignment at load time.
- Convert GGUF tensor metadata into the canonical packed layout once.
- Do not repack or fully dequantize weights on the steady-state hot path.
- Keep any minimal/simple loader behavior out of strict real-inference mode.

### Tokenizer resolution precedence

1. explicit tokenizer override;
2. tokenizer embedded in or referenced by GGUF metadata, if supported by that model path;
3. sibling tokenizer assets next to the model, such as `tokenizer.json` and `tokenizer_config.json`;
4. fail in strict mode;
5. compatibility fallback only in non-strict mode, and only when recorded in the receipt.

Hardcoded GPT-2-style fallback is acceptable only as a compatibility/debug path, never as strict inference behavior.

## Kernel matrix

| Kernel/op | Why it matters | First implementation | Fast implementation |
| --- | --- | --- | --- |
| packed QK256/I2_S GEMV | decode throughput | scalar reference | AVX2 first, NEON next, optional AVX-512 |
| packed QK256/I2_S GEMM | prefill throughput | scalar reference | AVX2, NEON, optional AVX-512 |
| block unpack/dequant | parity/debug/offline inspection | scalar | optional SIMD helpers |
| RMSNorm | every layer | scalar | AVX2 + NEON |
| RoPE | every attention step | scalar | AVX2 + NEON |
| Q·K^T score path | attention | scalar | AVX2 + NEON where profitable |
| softmax + scale + mask | attention | scalar | vectorized where profitable |
| A·V path | attention | scalar | AVX2 + NEON |
| KV-cache append/read/stride helpers | decode latency | scalar | cache-aware CPU implementation |
| embedding gather | input path | scalar | cache-aware CPU implementation |
| logits/output head | every step | scalar | packed/vectorized if layout supports it |

Decode should be optimized before generalized prefill elegance. Prefill is GEMM-like; decode is GEMV-like and is usually more sensitive to memory layout, cache residency, and hidden fallbacks.

## Hardware lane policy

| Lane | Planning target | Rule |
| --- | --- | --- |
| scalar | all CPUs | correctness oracle and CI fallback; must be deterministic. |
| AVX2/FMA | mainstream x86-64 | first fast decode lane and baseline for current x86 systems. |
| AVX-512 | selected x86-64 hosts | optional; never required for a fast native CPU path. |
| NEON | arm64 | first Apple/Arm native CPU lane after AVX2 decode is proven. |

Do not make wider ISAs the only fast path. Dispatch should select them only after CPUID/feature probing and parity coverage.

## Canonical packed layout API target

`crates/bitnet-qk256-layout-core` should become the authority for QK256/I2_S geometry, alignment, block iteration, and row/column traversal. A future API can use this shape:

```rust
pub struct Qk256BlockView<'a> {
    // canonical packed bytes, scales, and block metadata
}

pub trait PackedWeightMatrix {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn row_blocks(&self, row: usize) -> &[Qk256BlockView<'_>];
}
```

Acceptance criteria:

- the model loader emits this layout directly;
- scalar and SIMD kernels consume this layout directly;
- steady-state inference does not duplicate repacking or reinterpret block geometry in multiple crates.

## PR-sized work items

### CPU-001: unify GGUF authority

**Touchpoints:** `crates/bitnet-models/src/formats/gguf/loader.rs`, `crates/bitnet-models/src/gguf_simple.rs`, `crates/bitnet-models/src/lib.rs`, optionally server model loading.

**Target API:**

```rust
pub fn load_gguf_model(path: &Path, opts: &LoadOptions) -> Result<LoadedBitNetModel>;
```

**Acceptance:** one canonical real-inference load path; quant/layout metadata validation; strict mode rejects unsupported or partial models.

### CPU-002: unify tokenizer authority

**Touchpoints:** tokenizer GGUF loader, GGUF tokenizer, auto/universal resolver, CLI tokenizer discovery.

**Target API:**

```rust
pub fn resolve_tokenizer(model_path: &Path, opts: &TokenizerOptions) -> Result<ResolvedTokenizer>;
```

**Acceptance:** deterministic precedence order; explicit receipt field for tokenizer source; strict mode fails instead of guessing.

### CPU-003: canonical packed layout crate

**Touchpoints:** `crates/bitnet-qk256-layout-core`, quantization dispatch, model qk256 helpers.

**Acceptance:** one block definition, one alignment contract, one iterator contract, and no duplicate steady-state repacking.

### CPU-004: scalar packed reference kernels

**Touchpoints:** quantization QK256 source and kernel FFI bridge.

**Target APIs:**

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

**Acceptance:** exact layout/pack fixtures; parity against debug dequant views; debug dequant helpers remain test-only.

### CPU-005: AVX2 decode-first GEMV

**Touchpoints:** AVX2 QK256 implementation, matmul dispatch, dispatch planner.

**Target API:**

```rust
pub unsafe fn qk256_gemv_avx2(
    w: &impl PackedWeightMatrix,
    x: &[f32],
    y: &mut [f32],
) -> Result<()>;
```

**Acceptance:** selected only on AVX2/FMA-capable hosts; parity with scalar; meaningful speedup on decode microbenchmarks.

### CPU-006: CPU transformer op lane

**Touchpoints:** CPU kernel modules, transformer crate, inference backend.

**Target APIs:**

```rust
pub fn rmsnorm_f32_inplace(x: &mut [f32], weight: &[f32], eps: f32);
pub fn apply_rope_inplace(q: &mut [f32], k: &mut [f32], pos: usize, cfg: &RopeCfg);
pub fn kv_append(cache: &mut KvCache, layer: usize, token: usize, k: &[f32], v: &[f32]) -> Result<()>;
```

**Acceptance:** deterministic layer/block parity; decode can run a real model path without generic fallback.

### CPU-007: strict mode and receipts

**Touchpoints:** backend selection, inference backends, receipts, CLI.

**Acceptance:** receipt records requested/selected backend and kernel, tokenizer source, fallback flag, and fallback reason; strict mode rejects hidden fallback; benchmark artifacts are machine-readable.

### CPU-008: NEON and optional AVX-512 widening

**Touchpoints:** new NEON/AVX-512 modules, dispatch tables, parity tests.

**Acceptance:** NEON parity/speedup on arm64; AVX-512 remains optional and benchmark-proven.

## Parity tolerances

| Tier | Examples | Policy |
| --- | --- | --- |
| bit/pack exact | block pack/unpack, metadata, tensor offsets | exact byte equality |
| kernel numeric parity | scalar packed vs AVX2/NEON packed | exact integer accumulation when possible; otherwise `rtol=1e-5`, `atol=1e-5` |
| layer outputs | transformer block fixtures | starting policy `rtol=1e-4`, `atol=1e-4` |
| logits/generation | top-k logits, greedy decode | compare top-1/top-k agreement and bounded drift; greedy temperature-0 fixtures should not diverge without an explained release-blocking issue |

## Benchmark profiles

| Profile | Purpose | Required dimensions |
| --- | --- | --- |
| micro | single kernel, synthetic blocks, controlled cache state | wall time, median, p95, bandwidth if relevant, selected kernel |
| layer | one transformer block at fixed shapes | per-op breakdown and parity |
| prefill | prompt-only throughput | prompt tokens/sec, prompt length, batch size, thread count |
| decode | steady-state generation | generated tokens/sec, latency p50/p95, KV-cache size, selected kernel |

Every benchmark receipt should include fallback status, CPU features, thread count, model id/path, prompt tokens, generated tokens, and selected kernel.

## CPU receipt target shape

```json
{
  "schema_version": "1.1.0",
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

Keep this compatible with the existing receipt schema family by adding optional CPU-specific fields rather than breaking current readers.

## Commands to standardize

```bash
cargo test --locked --workspace --no-default-features --features cpu
cargo test --locked -p bitnet-common --no-default-features --features cpu
cargo test --locked -p bitnet-quantization --release --no-default-features --features cpu
cargo bench --locked -p bitnet-quantization --bench qk256_gemv --features cpu
cargo bench --locked -p bitnet-kernels --bench kernel_benchmarks --features cpu
```

Future receipt-producing CLI shape:

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

## Implementation checklist

- [ ] Remove split-brain GGUF loading paths for real inference.
- [ ] Make tokenizer resolution explicit, deterministic, and strict-mode aware.
- [ ] Make `bitnet-qk256-layout-core` the packed-layout authority.
- [ ] Land scalar packed GEMV/GEMM reference kernels.
- [ ] Land AVX2 decode GEMV before prefill GEMM optimization.
- [ ] Add RMSNorm, RoPE, KV-cache, attention score/value, embedding, and output-head CPU helpers.
- [ ] Record requested vs selected backend/kernel and fallback reason in receipts.
- [ ] Fail strict mode on hidden fallback.
- [ ] Add microbench, layerbench, prefill bench, and decode bench receipts.
- [ ] Publish reproducible receipt JSON in CI/manual validation.
