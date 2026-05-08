# CUDA-ANSWER-012 BitNet I8_S CUDA Parity

## Summary

`CUDA-ANSWER-012` fixes the first generated-answer CPU/CUDA parity gap from
`CUDA-ANSWER-011`. The CUDA QK256 path now uses the BitNet.cpp-style I2_S x
I8_S activation-quantized dot when an inline I2_S scale is present, instead of
running a raw FP32 dot and applying the scale afterward.

The refreshed RTX 5070 Ti CUDA answer-corpus receipt still passes all five
deterministic cases with `selected_backend=nvidia-rtx-5070-ti-cuda`,
`runtime_api=cuda`, `fallback_used=false`, `selected_kernel=qk256_gemv_cuda`,
and `speedup_claim=false`.

The refreshed CPU/CUDA parity receipt improves from 0/5 exact case parity to
1/5 exact case parity. More importantly, generated token IDs and decoded text
now match for all five cases. The four remaining parity failures are top-k
logit-vector differences only.

## Root Cause

The CPU reference dispatch applies the BitNet.cpp I8_S activation path for
QK256 tensors when the GGUF inline scale is present:

```text
I2_S codes x quantized I8_S activations -> integer dot
subtract activation sum
divide by activation scale
multiply by inline weight scale
```

The previous CUDA path computed a raw FP32 QK256 dot and then applied the
inline scale outside the CUDA kernel. That was close enough for four visible
answers, but the `yes_no_water` prompt had a small first-token margin and
flipped from the CPU answer `No. Water is` to CUDA `Yes.`.

## Evidence

| Receipt | Purpose |
|---|---|
| `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-answer-corpus.json` | Refreshed RTX 5070 Ti CUDA answer-corpus run after the I8_S CUDA fix. |
| `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-answer-corpus.json` | Same-box 9950X3D AVX-512 CPU baseline from `CUDA-ANSWER-011`. |
| `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-vs-cuda-answer-parity.json` | Refreshed CPU/CUDA parity comparison after the I8_S CUDA fix. |

CUDA corpus result:

| Case | CUDA answer | Generated token IDs | Result |
|---|---|---|---|
| `math_2_plus_2` | `4` | `[220, 19, 128009]` | pass |
| `capital_france` | `Paris` | `[12366, 128009]` | pass |
| `repeat_colors` | `red blue green` | `[2579, 6437, 6307, 128009]` | pass |
| `say_ok` | `OK` | `[10619, 128009]` | pass |
| `yes_no_water` | `No. Water is` | `[2360, 13, 10164, 374]` | pass |

Parity result:

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

Case-level parity:

| Case | Generated token IDs | Decoded text | Top-k logits |
|---|---|---|---|
| `math_2_plus_2` | match | match | differ |
| `capital_france` | match | match | differ |
| `repeat_colors` | match | match | differ |
| `say_ok` | match | match | differ |
| `yes_no_water` | match | match | match |

The first remaining divergence is still `capital_france` at step 0 with
scope `logits_or_kernel`; the chosen token ID is `12366` on both CPU and CUDA.

## Claim Boundary

Allowed claims:

- The strict RTX 5070 Ti CUDA answer corpus still passes all five committed
  deterministic cases for the official Microsoft I2_S artifact.
- The five-case CPU AVX-512 versus RTX 5070 Ti CUDA comparison now has matching
  generated token IDs and decoded text.
- `yes_no_water` generated-answer parity is closed.

Not allowed:

- Exact CPU/CUDA top-k logit parity.
- Broad chat quality beyond the committed deterministic answer corpus.
- CUDA speedup or throughput superiority.
- Full CUDA residency for every transformer operation.

## Next Step

The next CUDA answer-readiness item should localize the remaining top-k-only
logit drift without changing the answer-quality or speed claim boundary. The
useful follow-up is an exact numeric parity probe for the first decode step,
starting with the remaining CUDA QK256/CPU accumulation and rounding details.
