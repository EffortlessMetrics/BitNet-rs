# CUDA-BITNET-PERF-004 Benchmark Qualification Review

## Summary

`CUDA-BITNET-PERF-004` reviews the current RTX 5070 Ti BitNet CUDA benchmark
evidence without upgrading any speed claim.

The reviewed evidence is useful:

- repeated strict ask runs compare the same official Microsoft I2_S GGUF on
  CPU AVX-512 and RTX 5070 Ti CUDA;
- repeated warm-session runs prove CUDA session reuse with model/tokenizer/CUDA
  context initialized once and QK256 weights uploaded once;
- both receipts preserve `fallback_used=false`, `qk256_gemv_cuda`, measured
  QK256 kernel time, and transfer byte counters.

The evidence is still baseline evidence. It is not yet an accepted speedup
claim.

## Receipt

Primary receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-004-benchmark-qualification.json
```

Inputs reviewed:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-prod-004-answer-path-benchmark.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-002-repeated-strict-ask.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-003-warm-session-benchmark.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-vs-cuda-answer-parity.json
```

## Decision

Result:

```text
speedup_claim=false
benchmark_qualified_speedup=false
qualification_decision.status=not_accepted
```

No profile is upgraded by this review.

## Reviewed Profiles

| Profile | Evidence | Decision | Blocking gap |
| --- | --- | --- | --- |
| `strict_ask_math_8` | 3 CPU AVX-512 runs and 3 CUDA runs, same model/tokenizer/prompt policy, generated answers match, fallback-free. | Not accepted | Transfer timing is not measured, strict ask power/thermal samples are incomplete, and no profile-specific threshold has been accepted. |
| `strict_cuda_warm_session_2_turns` | 3 CUDA warm-session runs with model/tokenizer/context loaded once and upload-once QK256 handles. | Not accepted | No same-profile CPU AVX-512 warm-session comparator is committed. |

## Baseline Evidence

Repeated strict ask:

```text
CPU AVX-512 median total ms: 18797.0
CUDA median total ms: 2136.0
observed median ratio: 8.8001
QK256 kernel time ms: 2977.406
host_to_device_bytes: 168376320
device_to_host_bytes: 172247040
```

Repeated warm session:

```text
CUDA runs: 3
turns per run: 2
CUDA median total session ms: 8038.352
CUDA median kernel time ms: 2036.619
CUDA median generated tok/s: 1.3684
```

## Claim Boundary

May claim:

- benchmark qualification review evidence exists for the current repeated
  strict ask and warm-session receipts;
- current BitNet CUDA benchmark receipts remain fallback-free and selected
  `nvidia-rtx-5070-ti-cuda`;
- repeated strict ask and warm-session receipts record QK256 kernel timing and
  transfer byte counters.

Must not claim:

- accepted CUDA speedup;
- broad chat quality;
- production server readiness;
- full transformer CUDA residency;
- dense regular-LLM CUDA proof as BitNet packed I2_S/QK256 proof.

## Next Work

The next performance item should add repeated same-model decode-profile
receipts for CPU AVX-512 and CUDA, then run another qualification review with
profile-specific acceptance thresholds and complete transfer timing plus
power/thermal context.
