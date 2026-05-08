# CUDA-PROD-004 Benchmark Baseline

## Summary

`CUDA-PROD-004` adds a governed strict RTX 5070 Ti answer-path benchmark
receipt after execution-residency coverage. The receipt compares the same
official Microsoft BitNet I2_S artifact, explicit llama-bpe tokenizer
authority, `bitnetcpp-answer` prompt template, and deterministic math prompt on
the Windows 9950X3D CPU AVX-512 path and the RTX 5070 Ti CUDA path.

This is a benchmark baseline, not an accepted speedup claim. The receipt keeps
`speedup_claim=false` and records unmeasured timing fields and blocked long
profiles explicitly.

## Receipt

Primary receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-prod-004-answer-path-benchmark.json
```

Source receipts:

```text
target/bitnet/receipts/cuda-answer-readiness/strict-cpu-avx512-ask-math-benchmark.json
target/bitnet/receipts/cuda-answer-readiness/strict-cuda-ask-math-benchmark.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-answer-corpus.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-answer-corpus.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-vs-cuda-answer-parity.json
```

## Measured Strict Ask Profile

Profile:

```text
strict_ask_math_8
```

Prompt:

```text
What is 2+2? Answer with only the number.
```

The CPU AVX-512 and CUDA receipts both generated `4` with matching generated
token IDs and fallback-free execution.

| Backend | Runtime | Kernel | Total ms | First token ms | Tokens/sec | Fallback |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `amd-9950x3d-cpu-avx512` | `cpu` | `i2_s-avx512-reference` | 19410.0 | 17460.0 | 0.1546 | false |
| `nvidia-rtx-5070-ti-cuda` | `cuda` | `qk256_gemv_cuda` | 1833.0 | 1645.0 | 1.6367 | false |

The observed CPU-total-ms / CUDA-total-ms ratio is recorded in the JSON receipt
as baseline evidence only. It is not an accepted speedup claim.

## Existing Corpus Evidence

The receipt links the existing governed answer-corpus and CPU/CUDA generated
answer parity receipts:

- CPU AVX-512 answer corpus: pass.
- RTX 5070 Ti CUDA answer corpus: pass.
- CPU/CUDA generated answer parity: pass for the deterministic corpus, with
  top-k parity still outside this benchmark claim.

## Blocked Long Profile

The attempted CPU AVX-512 `prefill_512_decode_128` phase benchmark timed out
after 1800 seconds before writing profile receipts. `CUDA-PROD-004` records
that profile as `blocked_timeout` and holds the matching CUDA long profile as
`not_run` until the same-profile CPU baseline exists.

This is intentional: the missing long profile does not weaken the receipt, and
it does not get converted into speed evidence.

## Claim Boundary

Allowed:

- Strict RTX 5070 Ti CUDA answer-path timing is measured for the deterministic
  ask profile against a same-model CPU AVX-512 receipt.
- CUDA used `selected_backend=nvidia-rtx-5070-ti-cuda`,
  `runtime_api=cuda`, `kernel=qk256_gemv_cuda`, and `fallback_used=false`.
- The receipt exposes timing splits, residency boundaries, and unavailable
  measurement fields.

Not allowed:

- CUDA speedup.
- Broad chat quality beyond deterministic answer receipts.
- Full transformer CUDA residency.
- Production server readiness.
- Separate CUDA context initialization, weight-upload timing, kernel-time, or
  host/device transfer timing claims.

## Next Work

The next benchmark-quality work is to make the long profile tractable and add
separate timing for CUDA context initialization, weight upload, kernel time,
and host/device transfer. Only after same-model, fallback-free profiles cover
the accepted benchmark set should this lane consider upgrading
`speedup_claim`.
