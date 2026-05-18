# BitNet CPU AVX2 Status

## Current claim boundary

The CPU AVX2 BitNet hot-path campaign is in planning. The repository has merged
rails for strict loader/tokenizer authority, canonical QK256 layout, scalar truth
kernels, AVX2 dispatch, CPU decode, receipts, and answer-parity diagnostics, but
this status page does not yet claim that the real inline-scale BitNet production
path uses an optimized scaled I2_S x I8_S AVX2 kernel.

## Official Microsoft BitNet I2_S/QK256 artifact

| Area | Status | Evidence requirement before promotion |
| --- | --- | --- |
| Scalar packed path | Correctness oracle | Preserve scalar tests and generated-token receipts. |
| AVX2 no-scale F32 QK256 GEMV | Separate kernel family | Must not substitute for inline-scale I2_S x I8_S proof. |
| AVX2 scaled I2_S x I8_S GEMV | Candidate / unproven until counters and kernel wiring land | Receipts must show selected scaled AVX2 kernel and scaled AVX2 invocation count greater than zero. |
| Strict fallback | Required fail-closed behavior | Requested AVX2 in strict mode must error rather than silently select scalar or diagnostic execution. |
| Answer corpus | Must remain scalar-vs-AVX2 parity gated | Generated token IDs must match or divergence evidence blocks optimization promotion. |
| Long decode | Planned | Greedy deterministic parity profiles for 16, 32, and 128 generated tokens. |
| Performance | Exact profiles only | Phase receipts for micro, prefill, first-token, decode, and warm-session profiles before any profile promotion. |
| Server / GPU / NPU | Not claimed | Out of scope for this campaign. |

## Required hot-path receipt counters

Future hot-path proof receipts must expose:

```text
qk256_f32_scalar_gemv_invocations
qk256_f32_avx2_gemv_invocations
qk256_i8s_scaled_scalar_invocations
qk256_i8s_scaled_avx2_invocations
qk256_flat_bytes_extracted_count
qk256_input_rows_materialized_count
qk256_output_rows_allocated_count
qk256_tensor_to_vec_count
```

The first runtime PR after this documentation rail is diagnostic only: it records
which path actually runs today and makes no speed claim.

## Promotion policy

Promote support profile by profile, not globally. A row may move from candidate
to ready only when strict receipts prove:

1. real GGUF loader authority;
2. strict tokenizer authority;
3. selected scaled AVX2 kernel where inline scale is present;
4. `fallback_used=false`;
5. scalar-vs-AVX2 generated-token parity; and
6. phase timing good enough for that exact profile.
