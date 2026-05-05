# BitNet CPU Path Plan

## Purpose

This document turns the CPU-path investigation into an implementation contract. It is
intended to keep follow-up PRs aligned around one coherent CPU inference lane instead
of several partially connected loader, tokenizer, layout, kernel, transformer, and
receipt paths.

The plan is documentation-first by design: it names the authority boundaries,
acceptance checks, receipt fields, and PR slices that future code changes must satisfy.
The companion machine-readable backlog lives in `docs/bitnet/cpu_path_plan.yml`.

## Executive Summary

The useful CPU path is not one missing function. It is three systems that must become
one strict, measurable lane:

1. **Model and tokenizer authority**: GGUF loading, model-family metadata, tokenizer
   resolution, and strict-mode fallback policy must converge on deterministic entry
   points.
2. **Packed quantized kernel authority**: QK256/I2_S layout types, scalar reference
   kernels, ISA-specific kernels, and dispatch must consume one packed layout without
   steady-state dequantization or repacking.
3. **Real transformer execution**: packed matmul is necessary but insufficient; decode
   needs RMSNorm, RoPE, attention score/value paths, KV-cache helpers, embedding
   gather, output head, batching, and prefill/decode scheduling.

The shortest reliable order is:

1. make loader, tokenizer, and packed layout authoritative and strict;
2. make scalar packed reference kernels correct and receipt-backed;
3. make AVX2 decode-first kernels fast enough for real models;
4. widen to NEON and optional AVX-512 only after AVX2 and scalar parity are proven.

## Canonical CPU Inference Lane

All CPU inference work should be wired toward this path:

```text
CLI/server request
  -> bitnet-common backend selection
  -> bitnet-inference backend adapter
  -> canonical GGUF loader
  -> canonical tokenizer resolver
  -> QK256/I2_S packed layout selection
  -> prefill kernel dispatch
  -> decode kernel dispatch
  -> scalar / AVX2 / AVX-512 / NEON implementation
  -> CPU transformer ops
  -> receipts and benchmark artifacts
```

The important rule is that these stages must not silently disagree. If a strict run
requests AVX2 QK256 decode, but the runtime selected scalar or dequantized reference
compute, the run must fail or record an explicit non-strict fallback.

## Current Repo Surfaces to Reconcile

| Area | Primary files | Build-on rule |
|---|---|---|
| GGUF loading | `crates/bitnet-models/src/formats/gguf/loader.rs`, `crates/bitnet-models/src/gguf_simple.rs` | One canonical real-GGUF path for inference; simple/minimal paths are compatibility tooling only. |
| Tokenizer authority | `crates/bitnet-tokenizers/src/gguf_loader.rs`, `crates/bitnet-tokenizers/src/gguf_tokenizer.rs`, `crates/bitnet-tokenizers/src/auto.rs`, `crates/bitnet-tokenizers/src/universal.rs`, `crates/bitnet-cli/src/tokenizer_discovery.rs` | Deterministic precedence with no hardcoded fallback in strict mode. |
| Packed layout | `crates/bitnet-qk256-layout-core/src/lib.rs`, `crates/bitnet-quantization/src/qk256_dispatch.rs`, model-side QK256 helpers | Layout geometry, alignment, row iteration, and tensor interpretation come from one authority. |
| Scalar packed kernels | `crates/bitnet-quantization/src/i2s_qk256.rs` | Truth path for CI, parity, fuzzing, and fallback receipts. |
| AVX2 packed kernels | `crates/bitnet-quantization/src/i2s_qk256_avx2.rs` | Decode GEMV first; prefill GEMM second. |
| Kernel dispatch | `crates/bitnet-kernels/src/matmul_dispatch.rs`, `crates/bitnet-kernels/src/dispatch_planner.rs`, `crates/bitnet-kernels/src/dispatch_table.rs` | Dispatch by workload phase and runtime ISA, not by optimistic compile target. |
| Transformer CPU ops | `crates/bitnet-kernels/src/cpu/`, `crates/bitnet-transformer/src/lib.rs`, `crates/bitnet-inference/src/backends.rs` | RMSNorm, RoPE, attention, KV-cache, embeddings, and logits must be part of the CPU lane contract. |
| Strict mode and receipts | `crates/bitnet-common/src/strict_mode.rs`, `crates/bitnet-common/src/backend_selection.rs`, `crates/bitnet-receipts/**`, `crates/bitnet-cli/**` | Requested vs selected backend/kernel and fallback reason must be receipt-visible. |

## Loader and Tokenizer Authority

### Loader Rules

- Parse GGUF metadata once.
- Normalize model family, tensor names, RoPE parameters, GQA/head layout, and quant
  format at load time.
- Validate QK256/I2_S block geometry and alignment before exposing tensors to kernels.
- Prefer mmap/zero-copy-compatible tensor views where practical.
- Keep full-matrix dequantization out of steady-state inference.
- In strict mode, reject unsupported or partial models instead of falling back to a
  minimal loader.

### Tokenizer Resolution Precedence

Tokenizer resolution must be deterministic:

1. explicit tokenizer override;
2. tokenizer embedded in or referenced by GGUF metadata, when supported;
3. sibling tokenizer assets next to the model, such as `tokenizer.json` and
   `tokenizer_config.json`;
4. failure in strict mode.

Hardcoded GPT-2 or other compatibility fallbacks may remain only in non-strict tooling,
and must be recorded as fallback behavior when they are used.

## Packed Layout Contract

The CPU lane should treat packed weights as immutable execution data:

- one canonical block definition;
- one alignment contract;
- one row/column/block iteration API;
- one place where GGUF tensor metadata becomes executable layout;
- no duplicate repacking in steady-state inference;
- no dequantized dense tensor path for performance claims.

The layout-core crate should eventually expose the shared view/trait consumed by model
loading, quantization, kernels, dispatch, tests, and benches.

## Kernel Matrix

| Kernel target | Primary workload | First lane | Acceptance |
|---|---|---|---|
| Scalar packed GEMV/GEMM | correctness, CI, fallback | all CPUs | Exact layout parity and deterministic golden fixtures. |
| AVX2/FMA packed GEMV | decode | mainstream x86-64 | CPUID-gated, scalar parity, measurable speedup. |
| AVX2/FMA packed GEMM | prefill | mainstream x86-64 | Separate tiled prefill path after decode GEMV. |
| NEON packed GEMV/GEMM | decode, then prefill | arm64 | Scalar parity and arm64 receipts. |
| Optional AVX-512 packed GEMV/GEMM | decode and prefill | selected x86-64 | Optional dispatch only; never required for the first fast path. |
| Reference dequant + dense matmul | tests and diagnostics | all CPUs | Not valid for steady-state packed-kernel performance claims. |

The decode loop should be optimized before prefill because single-token GEMV,
KV-cache traffic, norm/RoPE overhead, and output-head latency dominate interactive
local inference.

## Minimum Decode-Critical CPU Ops

| Operation | Why it matters | First implementation | Fast implementation |
|---|---|---|---|
| Packed GEMV | one-token decode projections and output head | scalar packed | AVX2, NEON, optional AVX-512 |
| Packed GEMM | prompt prefill | scalar packed | AVX2, NEON, optional AVX-512 |
| Block unpack/dequant | debug and parity only | scalar | vectorized optional |
| RMSNorm | every layer | scalar | AVX2 + NEON |
| RoPE | every attention step | scalar | AVX2 + NEON |
| QK score | attention | scalar | AVX2 + NEON |
| Softmax/masking | attention | scalar | vectorized where profitable |
| AV step | attention | scalar | AVX2 + NEON |
| KV-cache append/read | decode | scalar | cache-aware CPU implementation |
| Embedding gather | input path | scalar | cache-aware CPU implementation |
| Logits/output head | every generated token | scalar | packed/vectorized when layout supports it |

## Strict Mode Contract

| Mode | Allowed | Not allowed |
|---|---|---|
| `auto` | scalar fallback, tokenizer discovery fallback, reference dequant fallback for diagnostics | fake success receipts |
| `strict` | only the requested loader, tokenizer, backend, kernel, and layout path | minimal-loader fallback, hardcoded tokenizer fallback, hidden scalar substitution, hidden dequantized steady-state inference |

Strict mode failures should include the requested path, selected path, and rejection
reason so receipts and logs are actionable.

## Receipt Shape for CPU Proofs

A CPU proof receipt must record both request and selection:

```json
{
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
    "format": "gguf",
    "family": "bitnet",
    "quant_format": "i2_s_qk256",
    "loader_mode": "strict"
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

These fields extend, rather than replace, the canonical BitNet receipt fields in
`docs/bitnet/BITNET_RECEIPT_FIELDS.md`.

## Parity Policy

| Tier | Example | Starting tolerance |
|---|---|---|
| Bit/pack exact | block pack/unpack, metadata, tensor offsets | exact byte equality |
| Kernel numeric parity | scalar packed vs AVX2/NEON packed | exact integer accumulation where possible, otherwise `rtol=1e-5`, `atol=1e-5` |
| Layer/model parity | layer outputs, logits, greedy tokens | `rtol=1e-4`, `atol=1e-4`, top-k/token agreement for logits and generation |

Greedy decode with fixed seed and temperature zero should match fixture prompts. Any
divergence is a release blocker until explained and receipt-recorded.

## Benchmark Profiles

| Profile | Purpose | Must record |
|---|---|---|
| `micro` | one kernel, synthetic blocks, controlled cache state | selected kernel, features, fallback flag, median/p95 |
| `layer` | one transformer block at fixed shapes | layer config, max/mean error, phase timings |
| `prefill` | prompt-only throughput | prompt tokens/sec, prompt length, batch size |
| `decode` | steady-state generation throughput | generated tokens/sec, p50/p95 latency, KV-cache state |

Reports must include wall time, selected backend/kernel, CPU feature set, thread count,
model identifier, prompt length, generation length, fallback status, and artifact path.

## PR-Sized Work Items

| ID | Title | Primary files | Acceptance |
|---|---|---|---|
| CPU-001 | Unify GGUF authority | `crates/bitnet-models/src/formats/gguf/loader.rs`, `crates/bitnet-models/src/gguf_simple.rs`, `crates/bitnet-models/src/lib.rs` | One canonical load path; strict mode rejects unsupported/partial models. |
| CPU-002 | Unify tokenizer authority | `crates/bitnet-tokenizers/src/gguf_loader.rs`, `crates/bitnet-tokenizers/src/gguf_tokenizer.rs`, `crates/bitnet-tokenizers/src/auto.rs`, `crates/bitnet-tokenizers/src/universal.rs`, `crates/bitnet-cli/src/tokenizer_discovery.rs` | Deterministic precedence; strict mode fails instead of guessing. |
| CPU-003 | Canonical packed layout crate | `crates/bitnet-qk256-layout-core/src/lib.rs`, `crates/bitnet-quantization/src/qk256_dispatch.rs`, model-side QK256 helpers | Loader emits and kernels consume one layout API without hot-path repacking. |
| CPU-004 | Scalar packed reference kernels | `crates/bitnet-quantization/src/i2s_qk256.rs`, FFI bridge files | Stable golden fixtures and scalar GEMV/GEMM truth path. |
| CPU-005 | AVX2 decode-first kernel | `crates/bitnet-quantization/src/i2s_qk256_avx2.rs`, kernel dispatch files | CPUID-gated AVX2 GEMV beats scalar and matches scalar parity. |
| CPU-006 | CPU transformer op lane | `crates/bitnet-kernels/src/cpu/`, `crates/bitnet-transformer/src/lib.rs`, `crates/bitnet-inference/src/backends.rs` | RMSNorm, RoPE, attention, KV-cache, embeddings, and logits run without generic hidden fallback. |
| CPU-007 | Strict mode and receipts | `crates/bitnet-common/src/backend_selection.rs`, `crates/bitnet-inference/src/backends.rs`, `crates/bitnet-receipts/**`, `crates/bitnet-cli/**` | Receipts record requested/selected kernel and fallback reason; strict mode rejects hidden fallback. |
| CPU-008 | NEON and optional AVX-512 widening | new `*_neon.rs`, new `*_avx512.rs`, dispatch tables/tests | Widen only after scalar and AVX2 decode are proven. |

## Standard Commands

These commands should be kept working as the CPU lane becomes stricter:

```bash
cargo test --locked --workspace --no-default-features --features cpu
cargo test --locked -p bitnet-common --no-default-features --features cpu
cargo test --locked -p bitnet-quantization --release --no-default-features --features cpu
cargo bench --locked -p bitnet-quantization --bench qk256_gemv --features cpu
cargo bench --locked -p bitnet-kernels --bench kernel_benchmarks --features cpu
```

A future receipt-producing command should converge on this shape:

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

## Related Docs

- `docs/bitnet/BITNET_MODEL_CONTRACT.md`
- `docs/bitnet/BITNET_QUANTIZATION_CONTRACT.md`
- `docs/bitnet/BITNET_KERNEL_MATRIX.md`
- `docs/bitnet/BITNET_RUNTIME_PHASES.md`
- `docs/bitnet/BITNET_RECEIPT_FIELDS.md`
- `docs/bitnet/BITNET_BENCHMARK_PROTOCOL.md`
- `docs/cpu-kernel-architecture.md`
