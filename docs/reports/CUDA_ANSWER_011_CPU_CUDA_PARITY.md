# CUDA-ANSWER-011 CPU/CUDA Answer Parity

## Summary

`CUDA-ANSWER-011` records the first same-box answer-corpus comparison between
the 9950X3D AVX-512 CPU path and the RTX 5070 Ti CUDA path for the official
Microsoft BitNet I2_S artifact.

Both backends pass the committed deterministic answer corpus. Exact CPU/CUDA
parity is not yet proven: the generic parity comparator records first-step
top-k logit differences for all five cases, and the `yes_no_water` case
generates different passing answers.

This report preserves the gap instead of weakening the gate.

## Evidence

| Receipt | Purpose |
|---|---|
| `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-answer-corpus.json` | 9950X3D AVX-512 CPU answer-corpus run. |
| `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-answer-corpus.json` | RTX 5070 Ti CUDA answer-corpus run from `CUDA-ANSWER-010`. |
| `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-vs-cuda-answer-parity.json` | Generic CPU/CUDA answer-corpus parity comparison. |

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
| `yes_no_water` | `Yes.` | `qk256_gemv_cuda` | pass |

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
    "passed": 0,
    "failed": 5,
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
| `yes_no_water` | `No. Water is` | `Yes.` | `generated_token_ids`, `decoded_text`, `logits_topk` |

For the first four cases, prompt token IDs, generated token IDs, decoded text,
backend identity, fallback status, and quality gates agree; the comparator still
flags the top-k logit vectors. For `yes_no_water`, both outputs pass the weak
yes/no quality gate, but generated IDs and decoded text differ.

## Repeat Check

The CUDA `yes_no_water` case was rerun after the parity failure and again
returned:

```text
Yes.
```

with `selected_backend=nvidia-rtx-5070-ti-cuda`, `runtime_api=cuda`,
`fallback_used=false`, and `selected_kernel=qk256_gemv_cuda`. The divergence is
not a stale committed receipt.

## Decision

`CUDA-ANSWER-011` proves that both CPU AVX-512 and RTX 5070 Ti CUDA pass the
committed answer corpus for the official answer-ready artifact. It does not
prove exact CPU/CUDA parity.

Allowed claim:

- Both same-box CPU AVX-512 and RTX 5070 Ti CUDA answer-corpus receipts pass the
  committed deterministic corpus for the official Microsoft I2_S artifact.

Not allowed:

- Exact CPU/CUDA generated-token parity.
- Exact CPU/CUDA top-k logit parity.
- A speedup claim.
- Broad chat quality beyond the committed corpus.

## Next Step

The next CUDA answer-readiness PR should localize the `logits_or_kernel`
divergence. Start with the QK256 CUDA/CPU numeric path for first-step logits,
because prompt token IDs and visible answers agree for four cases while top-k
logits differ.
