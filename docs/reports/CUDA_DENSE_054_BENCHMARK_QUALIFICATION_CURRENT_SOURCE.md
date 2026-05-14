# CUDA-DENSE-054 Benchmark Qualification Current-Source Review

## Scope

`CUDA-DENSE-054` reviews the current-source dense Qwen2.5 0.5B Q8_0 strict
RTX 5070 Ti CUDA evidence after the one-token, short-decode, and warm-session
proofs were refreshed.

This is a governed benchmark qualification review. It is not a new live CUDA
benchmark run and it does not upgrade any speed claim.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-benchmark-qualification-current-source.json
```

## Evidence Inputs

| Input | Receipt |
| --- | --- |
| Benchmark baseline | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-cuda-benchmark-baseline.json` |
| Repeated comparator | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-10/dense-gguf-qwen-repeated-comparator.json` |
| One-token current-source proof | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-13/dense-qwen25-q8-one-token-cuda.json` |
| Short-decode current-source proof | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-short-decode-current-source.json` |
| Warm-session current-source proof | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-warm-session-current-source.json` |

## Profile Decisions

| Profile | CPU mean total ms | CUDA mean total ms | H2D envelope ms | D2H logits ms | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `one_token` | 2872.8428 | 3978.5710 | 8071.3571 | 0.9181 | Not accepted |
| `short_decode_8` | 3528.0687 | 4199.9896 | 4319.3893 | 6.9240 | Not accepted |
| `warm_session_3_turns` | 4596.1352 | 5034.9288 | 3936.0470 | 24.5661 | Not accepted |

## Qualification Decision

No dense Qwen CUDA profile is speedup-qualified.

Blocked requirements:

- CUDA mean total time remains slower than the same-artifact CPU reference mean
  for every reviewed profile.
- Pure host-to-device CUDA event copy timing remains unmeasured; the current H2D
  values are model-load envelopes that include non-transfer overhead.
- No profile-specific speedup threshold has been accepted.

## Claim Boundary

May claim:

- dense Qwen benchmark qualification consumed the refreshed current-source
  one-token, short-decode, and warm-session receipts;
- the reviewed profiles remain fallback-free dense regular-LLM CUDA evidence;
- each reviewed profile has an explicit not-accepted speed decision.

Must not claim:

- accepted dense Qwen CUDA speedup;
- `benchmark_qualified_speedup=true`;
- pure CUDA event H2D copy timing;
- full CUDA residency;
- server readiness;
- BitNet packed I2_S/QK256 proof from dense CUDA evidence.

## Validation

```text
cargo run --locked -p bitnet-bench-receipts --bin dense_qwen_cuda_benchmark_qualification_receipt --no-default-features -- --one-token-transfer ci\hardware\windows-9950x3d-rtx5070ti\2026-05-13\dense-qwen25-q8-one-token-cuda.json --short-decode-transfer ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-short-decode-current-source.json --warm-session-transfer ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-warm-session-current-source.json --receipt-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-benchmark-qualification-current-source.json
python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-benchmark-qualification-current-source.json
```
