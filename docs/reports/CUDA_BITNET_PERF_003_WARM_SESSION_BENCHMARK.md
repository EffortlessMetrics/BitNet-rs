# CUDA-BITNET-PERF-003 Warm-Session Benchmark

## Summary

`CUDA-BITNET-PERF-003` adds repeated strict CUDA warm-session benchmark evidence
for the Windows 9950X3D + RTX 5070 Ti lane. It reuses the already-proven strict
CUDA warm-session path: the model and tokenizer load once, the CUDA context is
initialized once, QK256 weight handles are uploaded once, and two deterministic
turns run with `fallback_used=false`.

This is CUDA warm-session baseline evidence, not an accepted speedup claim.
`speedup_claim=false`, `benchmark_qualified_speedup=false`, and
`full_cuda_residency_claimed=false` remain part of the receipt contract.

## Receipt

Primary receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-003-warm-session-benchmark.json
```

Source receipts:

```text
target/bitnet/receipts/cuda-bitnet-perf-003/cuda-warm-session-run-1.json
target/bitnet/receipts/cuda-bitnet-perf-003/cuda-warm-session-run-2.json
target/bitnet/receipts/cuda-bitnet-perf-003/cuda-warm-session-run-3.json
```

## Profile

Profile:

```text
strict_cuda_warm_session_2_turns
```

Prompts:

```text
What is 2+2? Answer with only the number.
Answer yes or no: is water wet?
```

All three runs generated `4` for the first turn and passed the quality gate for
both turns. Every run preserved:

```text
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
selected_kernel = qk256_gemv_cuda
fallback_used = false
model_loaded_once = true
tokenizer_loaded_once = true
cuda_context_initialized_once = true
qk256_weights_uploaded_once = true
per_token_weight_upload = false
```

| Runs | Turns/run | Median total session ms | Median kernel ms | Median generated tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 2 | 8038.352 | 2036.619 | 1.3684 |

## CUDA Counters

The repeated warm-session receipt aggregates:

```text
qk256_gemv_cuda invocations: 27090
kernel_time_ms: 6102.457
host_to_device_bytes: 344770560
device_to_host_bytes: 352696320
median VRAM high-water bytes: 9969860608
```

This proves measured QK256 kernel time and activation/output transfer byte
accounting for repeated warm sessions. It does not prove full transformer CUDA
residency.

## Claim Boundary

Allowed:

- Repeated strict CUDA warm-session benchmark evidence exists for the exact
  `strict_cuda_warm_session_2_turns` profile.
- The warm-session path loads model/tokenizer/context once, reuses uploaded
  QK256 weight handles, emits per-turn/session receipts, and remains
  fallback-free.
- CUDA selected `nvidia-rtx-5070-ti-cuda`, used `qk256_gemv_cuda`, and recorded
  measured QK256 kernel time plus transfer bytes.

Not allowed:

- Accepted CUDA speedup.
- Broad chat quality.
- Full transformer CUDA residency.
- Production server readiness.
- Dense regular-LLM CUDA proof as BitNet packed QK256 proof.

## Next Work

The next performance work should add decode-profile repetitions and then run a
benchmark qualification review for exact profiles. Only a later reviewed receipt
should upgrade `speedup_claim`.
