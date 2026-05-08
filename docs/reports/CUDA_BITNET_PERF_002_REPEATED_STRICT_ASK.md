# CUDA-BITNET-PERF-002 Repeated Strict Ask Benchmark

## Summary

`CUDA-BITNET-PERF-002` adds repeated strict ask benchmark evidence for the
Windows 9950X3D + RTX 5070 Ti lane. It compares the same official Microsoft
BitNet I2_S GGUF, explicit llama-bpe tokenizer authority, `bitnetcpp-answer`
prompt template, and deterministic math prompt on CPU AVX-512 and RTX 5070 Ti
CUDA.

This is benchmark qualification evidence, not an accepted speedup claim.
`speedup_claim=false` remains part of the receipt contract.

## Receipt

Primary receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-002-repeated-strict-ask.json
```

Source receipts:

```text
target/bitnet/receipts/cuda-bitnet-perf-002/cpu-avx512-strict-ask-run-1.json
target/bitnet/receipts/cuda-bitnet-perf-002/cpu-avx512-strict-ask-run-2.json
target/bitnet/receipts/cuda-bitnet-perf-002/cpu-avx512-strict-ask-run-3.json
target/bitnet/receipts/cuda-bitnet-perf-002/cuda-strict-ask-run-1.json
target/bitnet/receipts/cuda-bitnet-perf-002/cuda-strict-ask-run-2.json
target/bitnet/receipts/cuda-bitnet-perf-002/cuda-strict-ask-run-3.json
```

## Profile

Profile:

```text
strict_ask_math_8
```

Prompt:

```text
What is 2+2? Answer with only the number.
```

All six runs generated `4`, preserved matching generated token IDs, and recorded
`fallback_used=false`.

| Backend | Runs | Median total ms | Median tok/s | Kernel/time evidence |
| --- | ---: | ---: | ---: | --- |
| `amd-9950x3d-cpu-avx512` | 3 | 18797.0 | 0.1596 | `i2_s-avx512-reference` |
| `nvidia-rtx-5070-ti-cuda` | 3 | 2136.0 | 1.4045 | `qk256_gemv_cuda`, measured kernel time and transfer bytes |

Observed median CPU-total-ms / CUDA-total-ms ratio:

```text
8.8001
```

That ratio is recorded as baseline evidence only. It does not upgrade
`speedup_claim`.

## CUDA Counters

The repeated CUDA receipt aggregates:

```text
qk256_gemv_cuda invocations: 13230
kernel_time_ms: 2977.406
host_to_device_bytes: 168376320
device_to_host_bytes: 172247040
```

This proves QK256 timing and transfer byte accounting for the repeated strict
ask profile. It does not prove full transformer CUDA residency.

## Claim Boundary

Allowed:

- Repeated strict ask benchmark evidence exists for the exact
  `strict_ask_math_8` profile.
- CPU AVX-512 and RTX 5070 Ti CUDA use the same model, tokenizer, prompt
  template, question, and deterministic sampling policy.
- CUDA selected `nvidia-rtx-5070-ti-cuda`, used `qk256_gemv_cuda`, and recorded
  measured QK256 kernel time plus transfer bytes.

Not allowed:

- Accepted CUDA speedup.
- Broad chat quality.
- Full transformer CUDA residency.
- Production server readiness.
- Dense regular-LLM CUDA proof as BitNet packed QK256 proof.

## Next Work

The next performance work should define the benchmark qualification review for
specific profiles and then add warm-session and decode-profile repetitions.
Only a later reviewed receipt should upgrade `speedup_claim` for an exact
profile.
