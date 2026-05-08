# CUDA-ANSWER-011 CPU/CUDA Answer Parity

## Summary

`CUDA-ANSWER-011` records the first same-box answer-corpus comparison between
the 9950X3D AVX-512 CPU path and the RTX 5070 Ti CUDA path for the official
Microsoft BitNet I2_S artifact.

Both backends pass the committed deterministic answer corpus. The original
`CUDA-ANSWER-011` receipt preserved a real parity gap: first-step top-k logit
differences for all five cases, plus a `yes_no_water` generated-answer
divergence. The follow-up `CUDA-ANSWER-012` receipt refresh closes the generated
token divergence by applying BitNet I8_S activation semantics inside CUDA
QK256, while leaving exact top-k parity open for four cases.

This report preserves the remaining gap instead of weakening the gate.

## Evidence

| Receipt | Purpose |
|---|---|
| `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-answer-corpus.json` | 9950X3D AVX-512 CPU answer-corpus run. |
| `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-answer-corpus.json` | RTX 5070 Ti CUDA answer-corpus run refreshed by `CUDA-ANSWER-012`. |
| `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-vs-cuda-answer-parity.json` | Generic CPU/CUDA answer-corpus parity comparison refreshed by `CUDA-ANSWER-012`. |

## CPU Corpus Result

The CPU AVX-512 corpus receipt records `quality_summary.passed = 5`,
`quality_summary.failed = 0`, `quality_summary.timeout = 0`, and
`quality_summary.not_run = 0`.

| Case | CPU answer | Kernel | Result |
|---|---|---|---|
| `math_2_plus_2` | `4` | `i2_s-avx512-reference` | pass |
| `capital_france` | `Paris` | `i2_s-avx512-reference` | pass |
| `repeat_colors` | `red blue green` | `i2_s-avx512-reference` | pass |
| `say_ok` | `OK` | `i2_s-avx512-reference` | pass |
| `yes_no_water` | `No. Water is` | `i2_s-avx512-reference` | pass |

The CPU receipt records:

- `requested_backend = cpu`
- `selected_backend = cpu-rust`
- `runtime_api = cpu`
- `fallback_used = false`
- `selected_kernel = i2_s-avx512-reference`

## CUDA Corpus Result

The CUDA corpus receipt records `quality_summary.passed = 5`,
`quality_summary.failed = 0`, `quality_summary.timeout = 0`, and
`quality_summary.not_run = 0`.

| Case | CUDA answer | Kernel | Result |
|---|---|---|---|
| `math_2_plus_2` | `4` | `qk256_gemv_cuda` | pass |
| `capital_france` | `Paris` | `qk256_gemv_cuda` | pass |
| `repeat_colors` | `red blue green` | `qk256_gemv_cuda` | pass |
| `say_ok` | `OK` | `qk256_gemv_cuda` | pass |
| `yes_no_water` | `No. Water is` | `qk256_gemv_cuda` | pass |

The CUDA receipt records:

- `requested_backend = nvidia-rtx-5070-ti-cuda`
- `selected_backend = nvidia-rtx-5070-ti-cuda`
- `runtime_api = cuda`
- `fallback_used = false`
- `selected_kernel = qk256_gemv_cuda`

## Parity Result

The parity receipt records:

```json
{
  "artifact_kind": "bitnet_answer_corpus_parity",
  "summary": {
    "passed": 1,
    "failed": 4,
    "total": 5
  }
}
```

First recorded divergence:

| Field | Value |
|---|---|
| Case | `capital_france` |
| Kind | `logits_topk` |
| Scope | `logits_or_kernel` |
| Step | `0` |
| CPU chosen token | `12366` |
| CUDA chosen token | `12366` |

Case-level parity outcome:

| Case | CPU answer | CUDA answer | Failed rules |
|---|---|---|---|
| `math_2_plus_2` | `4` | `4` | `logits_topk` |
| `capital_france` | `Paris` | `Paris` | `logits_topk` |
| `repeat_colors` | `red blue green` | `red blue green` | `logits_topk` |
| `say_ok` | `OK` | `OK` | `logits_topk` |
| `yes_no_water` | `No. Water is` | `No. Water is` | none |

For all five cases, prompt token IDs, generated token IDs, decoded text, backend
identity, fallback status, and quality gates agree. Four cases still flag the
top-k logit vectors, so exact CPU/CUDA top-k parity remains open.

## Follow-Up Fix

`CUDA-ANSWER-012` found that the CUDA QK256 path applied the inline I2_S scale
after a raw FP32 dot product, while the CPU reference path used the BitNet.cpp
I2_S x I8_S activation-quantized integer dot when the inline scale was present.
The refreshed CUDA receipt now computes that I8_S activation path inside the
CUDA QK256 kernel and passes the same visible answer and generated token IDs as
the CPU receipt for all five corpus cases.

## Decision

`CUDA-ANSWER-011` proves that both CPU AVX-512 and RTX 5070 Ti CUDA pass the
committed answer corpus for the official answer-ready artifact. After
`CUDA-ANSWER-012`, it also records exact generated-token and decoded-text parity
for the five committed cases. It does not prove exact top-k logit parity.

Allowed claim:

- Both same-box CPU AVX-512 and RTX 5070 Ti CUDA answer-corpus receipts pass the
  committed deterministic corpus for the official Microsoft I2_S artifact.

Not allowed:

- Exact CPU/CUDA top-k logit parity.
- A speedup claim.
- Broad chat quality beyond the committed corpus.

## Next Step

The next CUDA answer-readiness PR should localize the remaining `logits_or_kernel`
divergence for the four top-k-only cases. Do not change claim boundaries:
generated-token parity for this five-case corpus is now receipt-backed, but
exact top-k parity and speed remain unclaimed.
