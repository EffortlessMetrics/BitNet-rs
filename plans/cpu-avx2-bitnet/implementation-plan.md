# CPU AVX2 BitNet Hot-Path Implementation Plan

## Operating rules

1. Work one tracker item per PR.
2. Keep scalar packed QK256/I2_S execution as the correctness oracle.
3. Keep strict mode fail-closed: requested AVX2 must not silently select scalar,
   dequantized, diagnostic, mock, or reference-only execution.
4. Keep receipt fields explicit for requested/selected backend, requested/selected
   kernel, runtime API, kernel family, fallback state, model authority, tokenizer
   authority, hot-path counters, and phase timing when performance is discussed.
5. Do not change tokenizer policy, prompt policy, or scalar semantics in this
   campaign unless a later spec explicitly supersedes this one.
6. Do not claim speed until exact-profile phase receipts have been reviewed.

## Implementation sequence

### CPU-AVX2-HOTPATH-000 -- docs/spec/plan/tracker rails

Add this plan, the behavior spec, the status document, and tracker entries for
the new campaign slice. This PR is documentation-only and makes no runtime claim.

Validation:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
git diff --check
```

### CPU-AVX2-HOTPATH-001 -- hot-path counters

Add counters that prove which QK256 path actually runs today:

- `qk256_f32_scalar_gemv_invocations`
- `qk256_f32_avx2_gemv_invocations`
- `qk256_i8s_scaled_scalar_invocations`
- `qk256_i8s_scaled_avx2_invocations`
- `qk256_flat_bytes_extracted_count`
- `qk256_input_rows_materialized_count`
- `qk256_output_rows_allocated_count`
- `qk256_tensor_to_vec_count`

Receipts must distinguish no-scale F32 GEMV from the real inline-scale
I2_S x I8_S path. This PR must not change math or make a speed claim.

### CPU-AVX2-HOTPATH-002 -- hot-path receipt validation

Make receipt validation fail hidden fallback cases:

- requested AVX2 selected scalar in strict mode;
- selected kernel says AVX2 but AVX2 invocation count is zero;
- inline-scale BitNet path records only no-scale F32 AVX2;
- `fallback_used=false` but counters show scalar substitution; or
- tensor materialization exceeds the audited boundary for a hot-path proof run.

### CPU-AVX2-HOTPATH-003 -- scaled I2_S x I8_S fixtures

Add scalar-grounded fixtures for rows `1, 2, 7, 32`, columns `256, 257, 300,
512, 513, 1024`, code patterns including code `3` and tails, activation
patterns, repeated-run determinism, and representative finite `weight_scale`
values. Fixtures compare to `gemv_qk256_bitnet_i8s_scaled` and must mirror the
scalar function exactly.

### CPU-AVX2-HOTPATH-004 -- scaled AVX2 GEMV

Add `gemv_qk256_bitnet_i8s_scaled_avx2` behind x86_64 AVX2/FMA feature gates and
runtime gating. The function must not fallback internally. Dimension checks,
tail handling, and `weight_scale` validation must match scalar behavior.

### CPU-AVX2-HOTPATH-005 -- explicit scaled kernel selection

Add stable kernel IDs for scaled scalar and scaled AVX2 I8_S GEMV, selection
metadata, strict fallback errors, and non-strict fallback receipts.

### CPU-AVX2-HOTPATH-006 -- transformer dispatch wiring

Route the inline-scale BitNet QK256 transformer branch through the scaled AVX2
selector when AVX2 is requested or auto-selected. Strict AVX2 answer-corpus
receipts must show the scaled AVX2 kernel, scaled AVX2 invocation count greater
than zero, scaled scalar count equal to zero, and `fallback_used=false`.

### CPU-AVX2-HOTPATH-007 -- packed-view/materialization cleanup

Cache parsed QK256 layout and flattened packed bytes at model load, expose an
immutable packed tensor view, use flat input/output buffers, and prove reduced
materialization counters without changing generated token IDs.

### CPU-AVX2-HOTPATH-008 -- reusable CPU decode workspace

Add a reusable workspace for activation I8 buffers, output F32 buffers, optional
QK256 scratch, attention scratch, and logits scratch. Receipts must show reuse
and memory high-water information when available.

### CPU-AVX2-HOTPATH-009 -- strict phase timing profiles

Add strict AVX2 phase timing profiles for micro QK256 scaled GEMV, layer-0
decode, prefill 128, prefill 512, first token, decode 32, decode 128, and warm
three-turn sessions. These receipts are evidence only; they do not by themselves
promote a speed claim.

### CPU-AVX2-HOTPATH-010 -- exact-profile performance review

Review scalar, AVX2, and previous-CPU timings profile by profile. Promote only
exact profiles with acceptable evidence. If AVX2 is slower, record the blocker
and next target.

### CPU-AVX2-HOTPATH-011 -- answer corpus v2

Expand the BitNet CPU answer corpus across math, factual, copy/repeat, yes/no,
short extraction, format following, multi-token continuation, stop-token, and
prompt-conditioning categories. Keep broad chat quality out of scope.

### CPU-AVX2-HOTPATH-012 -- long-decode deterministic parity

Add greedy deterministic scalar-vs-AVX2 parity for 16, 32, and 128 generated
tokens with fixed seeds where applicable. Receipts must record prompt token IDs,
generated token IDs, first divergence, top-k evidence when available, and
`fallback_used=false`.

### CPU-AVX2-HOTPATH-013 -- prefill path optimization

Optimize prefill separately from decode GEMV. Acceptable directions include
batched GEMV, tiled GEMM if needed, conservative threading, and no hot-path
weight dequantization. Receipts must prove prefill profile effects.

### CPU-AVX2-HOTPATH-014 -- non-QK256 CPU op audit

Measure RMSNorm, sub-layernorm, RoPE, QK score, softmax/masking, AV, KV
append/read, output head/logits, and sampling so the next optimization target is
chosen from evidence.

### CPU-AVX2-HOTPATH-015 -- user-facing AVX2 status

Publish the exact support status for the official Microsoft BitNet I2_S artifact:
scalar correctness oracle, AVX2 scaled I8_S readiness, answer corpus state,
long-decode state, exact speed profiles, and non-claims.

## Default validation bundle for runtime PRs

Most runtime PRs should run the scoped bundle below in addition to the item
specific commands when their touched crates support it:

```bash
cargo fmt --all -- --check
cargo test --locked -p bitnet-quantization --no-default-features --features cpu,avx2 --test qk256_avx2_parity_tests
cargo test --locked -p bitnet-quantization --no-default-features --features cpu,avx2 i2s_qk256 --lib
cargo test --locked -p bitnet-qk256-dispatch --no-default-features --features cpu
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli answer_corpus
cargo run --locked -p xtask --no-default-features -- campaign check cpu-proof
git diff --check
```

Performance PRs also add a no-run benchmark build and JSON receipt validation
for any new receipt artifact.
