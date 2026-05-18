# BitNet CPU AVX2 Status

This page records the current claim boundary for the CPU AVX2 BitNet I2_S/QK256
hot-path campaign. It is intentionally narrower than general CPU inference.

## Current status

| Area | Status | Notes |
| --- | --- | --- |
| Official Microsoft BitNet I2_S/QK256 scalar | correctness oracle | Existing scalar packed path remains the reference for AVX2 parity. |
| Existing no-scale QK256 AVX2 GEMV | proven helper lane | It is not proof of inline-scale BitNet I2_S x I8_S AVX2 execution. |
| Scaled I2_S x I8_S AVX2 hot path | planned | The next runtime item must first record counters that prove what executes today. |
| Strict loader/tokenizer authority | required | Hot-path proof must keep real GGUF loading and strict tokenizer source in receipts. |
| Hidden fallback | forbidden | Strict AVX2 proof must fail closed instead of warning-only scalar/dequant fallback. |
| Answer corpus parity | required | Scalar-vs-AVX2 generated token IDs must remain stable or divergence must block promotion. |
| Performance | unpromoted | No AVX2 speedup is claimed until exact-profile phase receipts are reviewed. |
| Server, GPU, NPU, dense SLM | out of scope | This campaign does not promote those proof families. |

## Promotion checklist

A CPU AVX2 BitNet profile can be promoted only after receipts show:

1. `requested_backend=cpu` and `selected_backend=cpu-rust`;
2. requested and selected kernel IDs name the intended scaled path;
3. `fallback_used=false` and `fallback_reason=null`;
4. `loader_mode=real_gguf`, `quant_format=i2_s`, and a model SHA;
5. strict tokenizer source and prompt policy are recorded;
6. scaled I2_S x I8_S AVX2 invocation counters are positive;
7. scaled scalar, dequantized, diagnostic, mock, and no-scale F32 substitution
   counters do not contradict the selected AVX2 path;
8. scalar-vs-AVX2 generated-token parity passes for the governed corpus/profile;
9. phase timing exists for the exact promoted profile.

## Next evidence gap

The immediate evidence gap is not generic AVX2 optimization. The immediate gap
is proving whether strict real BitNet CPU inference runs an optimized scaled
I2_S x I8_S AVX2 kernel or only reaches scalar/no-scale helper paths. The next
runtime PR must add hot-path counters and receipt fields without changing math
or claiming speed.
