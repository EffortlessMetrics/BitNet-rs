# CPU AVX2 BitNet Hot-Path Implementation Plan

## Goal

Get the official Microsoft BitNet I2_S/QK256 model from strict CPU correctness
proof to production-grade strict AVX2 execution on the normal Rust CPU user
path. Production-grade means the receipts prove real optimized BitNet AVX2
execution, not an AVX2 label over scalar, diagnostic, dequantized, or no-scale
F32 helper execution.

## Non-negotiable rails

1. **Scalar is the oracle.** Every AVX2 kernel is compared against the canonical
   scalar packed path. Generated-token parity must be preserved or a divergence
   receipt must block promotion.
2. **Strict mode fails closed.** If strict requested AVX2 cannot run AVX2, the
   run errors. Warning-only scalar, dequantized, diagnostic, mock, or reference
   fallback is forbidden in proof runs.
3. **Receipts are mandatory.** Proof receipts must include requested and
   selected backend, requested and selected kernel, kernel family, runtime API,
   fallback status and reason, real GGUF model identity, quant format, and
   tokenizer source/strictness.
4. **Performance claims need phase receipts.** No speedup or throughput claim is
   allowed without model/tokenizer identity, workload shape, phase timings, CPU
   features, selected kernel, fallback status, and exact-profile review.
5. **Proof families stay separate.** CPU AVX2 BitNet I2_S/QK256 evidence does
   not promote CUDA, NPU, OpenVINO, Apple M4, dense SLM, Qwen, server, or broad
   chat claims.

## Technical issue to prove first

The existing AVX2 QK256 path covers the older no-scale F32-style GEMV. Real
BitNet I2_S inline-scale inference uses the BitNet.cpp-aligned scaled I2_S x
I8_S activation flow: quantize activations to I8_S, compute over packed I2_S
codes, then apply scale and correction. The transformer dispatch already takes
the inline-scale branch through `gemv_qk256_bitnet_i8s_scaled`; therefore the
first runtime proof must distinguish scaled I2_S x I8_S scalar, scaled I2_S x
I8_S AVX2, no-scale F32 scalar, and no-scale F32 AVX2 execution.

## PR sequence

| Item | Title | Purpose | Claim boundary |
| --- | --- | --- | --- |
| CPU-AVX2-HOTPATH-001 | `docs(cpu): add AVX2 BitNet hot-path implementation plan` | Encode this plan, spec, status page, and tracker rails. | Documentation only; no runtime behavior changes. |
| CPU-AVX2-HOTPATH-002 | `diag(cpu): record BitNet QK256 hot-path execution counters` | Emit counters for scaled I8S scalar/AVX2, F32 scalar/AVX2, flat-byte extraction, row materialization, output allocation, and tensor-to-vec conversion. | No math changes and no speed claim. |
| CPU-AVX2-HOTPATH-003 | `receipts(cpu): validate AVX2 hot-path counters` | Make hidden fallback invalid in strict AVX2 receipts. | Validation only. |
| CPU-AVX2-HOTPATH-004 | `test(cpu): add scaled I2S-I8S AVX2 parity fixtures` | Define rows, tails, code patterns, activations, scales, code3 behavior, and determinism before new SIMD work. | Fixtures compare to scalar semantics. |
| CPU-AVX2-HOTPATH-005 | `feat(cpu): add AVX2 scaled I2S-I8S QK256 GEMV` | Implement a runtime-gated AVX2 scaled I2_S x I8_S GEMV with no internal fallback. | No transformer wiring until direct parity passes. |
| CPU-AVX2-HOTPATH-006 | `feat(cpu): select scaled AVX2 QK256 kernel explicitly` | Add explicit scaled scalar and scaled AVX2 kernel IDs and strict fallback behavior. | Selection metadata only. |
| CPU-AVX2-HOTPATH-007 | `feat(cpu): route inline-scale BitNet QK256 through scaled AVX2` | Wire strict/auto inline-scale dispatch to the scaled AVX2 selector. | Generated-token parity must remain green. |
| CPU-AVX2-HOTPATH-008 | `perf(cpu): cache QK256 packed views for CPU dispatch` | Remove avoidable flat-byte extraction and per-call row/output materialization. | Before/after receipts required; no global speed claim. |
| CPU-AVX2-HOTPATH-009 | `perf(cpu): add reusable BitNet CPU decode workspace` | Reuse activation, output, attention, logits, and optional code scratch buffers. | No generated-token drift. |
| CPU-AVX2-HOTPATH-010 | `bench(cpu): add strict AVX2 phase timing profiles` | Emit micro, layer, prefill, first-token, decode, and warm-session phase receipts. | Timing evidence only; promotion waits for review. |
| CPU-AVX2-HOTPATH-011 | `docs(cpu): review AVX2 performance qualification` | Accept or reject exact-profile promotions from scalar/AVX2/comparator timings. | No global speedup claim. |
| CPU-AVX2-HOTPATH-012 | `test(cpu): add BitNet CPU answer corpus v2` | Broaden deterministic CPU prompts and classify scalar/AVX2 failures. | No broad chat claim. |
| CPU-AVX2-HOTPATH-013 | `test(cpu): add scalar-vs-AVX2 long decode parity` | Check 16/32/128-token greedy parity and record first divergence evidence. | Fallback must remain false. |
| CPU-AVX2-HOTPATH-014 | `perf(cpu): optimize BitNet QK256 prefill path` | Improve prefill with batched GEMV or tiled GEMM without dequant fallback. | Exact prefill receipts only. |
| CPU-AVX2-HOTPATH-015 | `diag(cpu): profile non-QK256 transformer CPU ops` | Rank RMSNorm, RoPE, attention, softmax, KV, logits, sampling, and other support-op bottlenecks. | Diagnostic report only. |
| CPU-AVX2-HOTPATH-016 | `docs(cpu): publish AVX2 BitNet support status` | Publish exact status after receipts and performance review. | Exact profile/status rows only. |

## Default validation bundle

Documentation-only items run:

```bash
cargo fmt --all -- --check
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
git diff --check
```

Runtime and validation items add the scoped CPU AVX2 tests named by the active
tracker item. Performance items also add receipt JSON validation for every new
receipt.

## Rollback

Each item should be revertible independently. Runtime items must keep scalar
semantics unchanged so reverting an optimization or selector leaves the scalar
truth lane intact.
