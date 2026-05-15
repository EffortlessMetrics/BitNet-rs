# Apple M4 BitNet Productization Campaign

Campaign ID: `apple-m4-bitnet-productization`

Status: active

## Objective

Move Apple M4 BitNet from fixed one-shot and fixed-warm proof into
operator-ready warm sessions, then chat and serve only after receipt-backed
correctness, determinism, timeout, streaming, and failure-mode gates pass.

## Why This Exists

`apple-m4-local-answer` proved one-shot BitNet answers and a fixed-prompt warm
route for the accepted Microsoft I2_S artifact. `apple-m4-bitnet-eval-and-
benchmark` added a 100-case BitNet eval report, one-shot/fixed-warm benchmark
reports, and advisory regression comparison.

That is still not enough to enable BitNet chat or serve. Operators need a warm
route that can run their own bounded prompt sets, prove repeated-prompt
determinism, expose slow/failing stages, and preserve exact model/tokenizer/
backend/fallback receipts.

## Scope Boundary

This is an M4 Mac mini BitNet productization lane. It uses only the accepted
Microsoft I2_S GGUF, the accepted external Microsoft tokenizer authority, and
the `bitnetcpp-answer` prompt authority for answer evidence.

Dense Qwen evidence, dense SLM timing, and dense server success are not BitNet
product evidence. This campaign does not claim full Apple Metal inference,
QK256, Neural Engine execution, MPSGraph inference, MacBook behavior, speedup,
broad BitNet quality, or broad Apple Silicon performance.

## End State

- `bitnet mac bitnet-warm` supports operator-provided repeated prompts while
  preserving the fixed proof prompt default.
- Warm-session receipts record prompt source, accepted model/tokenizer identity,
  backend/fallback status, generated text, token IDs, per-turn timing, aggregate
  timing, memory, and repeated-prompt determinism.
- Slow or failed BitNet warm runs expose progress, timeout, and partial-failure
  receipts.
- BitNet chat remains disabled until variable warm-session receipts prove reuse,
  determinism, timeout, and failure boundaries.
- BitNet serve remains out of scope until chat, streaming, request/response, and
  receipt-export semantics are separately proven.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-BITNET-PROD-001 | merged | PR #4946 allowed `bitnet mac bitnet-warm` to run operator-supplied repeated prompts while keeping the fixed proof prompt default. |
| M4-BITNET-PROD-002 | in_progress | The 2026-05-15 bounded M4 variable warm-session receipt was recorded; PR packaging is in progress. |
| M4-BITNET-PROD-003 | proposed | Add warm-session progress, timeout, partial-failure receipts, and repair guidance. |
| M4-BITNET-PROD-004 | proposed | Define and enforce the BitNet chat enablement gate before enabling chat. |

## Review Policy

Each PR owns one item. Parser, CLI, fixture, and receipt validation checks belong
in generic CI. Live M4 model runs, long warm-session soaks, and timing envelopes
belong in local, advisory, scheduled, or release lanes.

Every item must preserve `apple-m4-cpu-neon`, `fallback_used=false`, exact
accepted model SHA, accepted tokenizer SHA/authority, `bitnetcpp-answer` prompt
authority, generated text/token IDs when runtime evidence exists, and explicit
disabled chat/serve/Metal/QK256/Neural Engine/MPSGraph/MacBook/broad-claim
boundaries.
