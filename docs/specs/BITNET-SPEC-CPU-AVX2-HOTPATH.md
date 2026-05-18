# BITNET-SPEC-CPU-AVX2-HOTPATH

## Status

Draft implementation spec for the CPU AVX2 BitNet hot-path campaign.

## Purpose

The CPU AVX2 BitNet lane moves from correctness existing somewhere in the tree
to production hot-path proof. The official Microsoft BitNet I2_S/QK256 model is
not considered fully working on AVX2 CPU until the normal Rust CPU user path can
prove strict loader/tokenizer authority, selected AVX2 BitNet kernels, no hidden
fallback, scalar-vs-AVX2 parity, intelligible answer receipts, and profile-level
performance evidence.

## Target end state

A strict CPU AVX2 proof run for the official Microsoft BitNet I2_S/QK256 model
must demonstrate all of the following:

1. `requested_backend` is `cpu` and `selected_backend` is the Rust CPU backend.
2. GGUF loading is authoritative and records `loader_mode = "real_gguf"`.
3. Tokenizer resolution is strict, deterministic, and receipt-backed.
4. The selected kernel is an actual AVX2 BitNet kernel for the production path.
5. Inline-scale BitNet QK256 inference uses the scaled I2_S x I8_S path, not the
   no-scale F32 GEMV path as a substitute.
6. `fallback_used = false` and `fallback_reason = null` in strict proof runs.
7. Scalar-vs-AVX2 prompt token IDs and generated token IDs remain stable, or a
   divergence receipt captures the first divergence and blocks optimization
   promotion.
8. Performance claims are limited to exact profiles with phase receipts.

## Strict fallback rules

Strict mode fails closed. When the user requests AVX2 in strict mode, the run
must fail if AVX2 cannot execute the required production kernel. It must not
continue with scalar, dequantized, diagnostic, mock, reference-only, or no-scale
F32 execution while presenting the result as strict AVX2.

Non-strict mode may fall back only if the receipt records the requested kernel,
selected kernel, `fallback_used = true`, and a concrete fallback reason.

## Required receipt fields

Every proof receipt for this campaign must include the existing CPU proof fields
and the hot-path fields below when applicable:

```json
{
  "requested_backend": "cpu",
  "selected_backend": "cpu-rust",
  "requested_kernel": "...",
  "selected_kernel": "...",
  "kernel_family": "i2_s|qk256",
  "runtime_api": "cpu",
  "fallback_used": false,
  "fallback_reason": null,
  "model": {
    "loader_mode": "real_gguf",
    "quant_format": "i2_s",
    "sha256": "..."
  },
  "tokenizer": {
    "source": "...",
    "strict": true
  },
  "qk256_hot_path": {
    "scaled_i8s_scalar_invocations": 0,
    "scaled_i8s_avx2_invocations": 0,
    "f32_scalar_invocations": 0,
    "f32_avx2_invocations": 0,
    "flat_bytes_extracted_count": 0,
    "input_rows_materialized_count": 0,
    "output_rows_allocated_count": 0,
    "tensor_to_vec_count": 0
  }
}
```

Performance receipts must additionally include phase timing fields for model
load, tokenizer load, prompt rendering, prefill, first token, total decode, and
tokens per second where the profile supports them.

## Scalar parity gate

Scalar packed QK256/I2_S execution remains the correctness oracle. An AVX2
optimization PR must preserve scalar-generated token parity. If parity changes,
the PR must include divergence evidence and must not merge as a performance or
optimization promotion.

## Scaled I2_S x I8_S hot-path requirement

The real inline-scale BitNet path quantizes activations to I8_S, computes an
integer dot over packed I2_S data, and applies the scale/sum correction. The
no-scale F32 AVX2 GEMV path is a separate kernel family and must not be used as
a substitute for the production inline-scale path.

The campaign therefore requires distinct counters, kernel IDs, selectors, tests,
and receipts for:

- no-scale QK256 F32 scalar GEMV;
- no-scale QK256 F32 AVX2 GEMV;
- scaled I2_S x I8_S scalar GEMV; and
- scaled I2_S x I8_S AVX2 GEMV.

## Performance promotion requirements

A speed claim requires before/after receipts for at least one stable profile and
must state exactly which profile is promoted. Raw timing receipts alone are not a
global speedup claim. Accepted profile rows must record scalar timing, AVX2
timing, previous CPU timing where available, acceptance decision, and reason.

## Forbidden claims

This spec does not authorize claims about CUDA, NPU, OpenVINO, Intel Arc A770,
Apple M4, dense SLMs, Qwen, server readiness, broad chat quality, or all BitNet
models. It applies only to the official Microsoft BitNet I2_S/QK256 CPU AVX2
proof path unless a later spec explicitly extends it.

## Acceptance for the docs rail

The documentation rail is accepted when:

- this spec exists;
- the implementation plan exists;
- the user-facing status page exists;
- the `cpu-proof` tracker includes the docs rail and the next hot-path counter
  item;
- no runtime files are changed;
- `cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof`
  passes; and
- `git diff --check` passes.
