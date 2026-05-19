# CUDA-MODEL-015 Qwen3 Repeated Comparator Baseline

Date: 2026-05-19
Campaign item: CUDA-MODEL-015
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen3-0.6b-instruct-q8_0
Coverage row: `dense_qwen3_06b_q8_candidate`

## Purpose

Qwen3 is product CLI-ready for normal RTX 5070 Ti dense CUDA ask/chat paths and
has exact-profile non-streaming server readiness after CUDA-MODEL-014B. The next
performance step is not another product promotion. It is a repeated
same-artifact CPU/CUDA comparator baseline that can support a later
benchmark-qualification review.

This report queues that baseline and records why the existing Qwen3 benchmark
receipt is not enough to promote speed or benchmark-qualified status.

## Current Evidence

Existing Qwen3 benchmark review:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-benchmark-qualification.json
```

Key blockers from that receipt:

| Profile | CPU mean total ms | CUDA mean total ms | Runs per backend | Repeated evidence | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| `one_token` | 3865.0375 | 4752.5861 | 1 | false | not accepted |
| `short_decode_8` | 5638.2572 | 6222.1281 | 1 | false | not accepted |
| `warm_session_3_turns` | 6697.5988 | 6832.0021 | 1 | false | not accepted |

The existing review also records that pure CUDA event H2D copy timing is
unmeasured. Its H2D field is a model-load wall-clock envelope, not a pure
transfer timing.

## Required Baseline

CUDA-MODEL-015 must collect repeated comparator evidence for:

- `one_token`;
- `short_decode_8`;
- `short_decode_32`;
- `warm_session_3_turns`;
- `decode_128_from_warm_context`.

Every profile must use the same Qwen3 artifact:

```text
qwen3-0.6b-instruct-q8_0
SHA-256: 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
```

Every profile must record:

- CPU AVX-512 comparator timing;
- RTX 5070 Ti CUDA comparator timing;
- identical tokenizer, prompt policy, and generation policy;
- requested backend and selected backend;
- selected route `dense_regular_llm_cuda`;
- `fallback_used=false`;
- quality result;
- model load, tokenizer load, prompt render, tokenize, CUDA context init,
  weight upload, prefill, first-token, decode total, and steady decode timing;
- kernel time and launch count;
- H2D and D2H bytes and timing, or an explicit unmeasured-source blocker;
- VRAM high-water mark;
- power and temperature context.

## Non-Claims

This report does not claim:

- Qwen3 speedup;
- Qwen3 benchmark-qualified speed;
- Qwen3 full CUDA residency;
- broader Qwen3 server readiness than the CUDA-MODEL-014B exact profile;
- broad dense GGUF CUDA readiness;
- Qwen2.5 proof inheritance;
- BitNet packed I2_S/QK256 proof;
- runtime math, tokenizer, loader, kernel, or server behavior changes.

## Validation

```powershell
rtk cargo run --locked -p xtask --no-default-features -- check-model-coverage
rtk cargo test --locked -p bitnet-bench-receipts --no-default-features qwen3
rtk cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
rtk cargo run --locked -p xtask --no-default-features -- campaign generate --check
rtk git diff --check
```
