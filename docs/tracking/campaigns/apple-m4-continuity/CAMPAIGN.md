# Apple M4 Continuity Campaign

Campaign ID: `apple-m4-continuity`

Status: active

## Objective

Keep the M4 Mac mini useful as the local Apple Silicon dense-SLM appliance while BitNet artifact qualification happens elsewhere: improve resident Mac UX, add compact health checks, polish latency, record longer resident-session envelopes, and prepare blocked BitNet proof scaffolding without widening hardware claims.

## Why This Exists

The dense Qwen2.5 SLM path is the practical Apple M4 user-facing path today. It proves the Mac CLI, model cache, receipts, warm-session behavior, quality corpus, regression guardrails, and Apple CPU/NEON routing for a regular dense SLM.

That evidence does not prove BitNet, 1-bit / 1.58-bit kernels, QK256, Neural Engine execution, or full Apple Metal inference. BitNet artifact qualification belongs to the artifact/MacBook lanes until an accepted artifact exists; the M4 mini should then run strict local proof, not hunt large artifacts.

## End State

- `bitnet mac chat` runs multiple dense-SLM prompts in one resident Apple M4 CPU/NEON session.
- A compact Mac smoke command can verify cache health, tiny answer behavior, receipt validity, backend/fallback identity, and disk/cache status.
- Latency polish focuses on perceived local UX and is backed by scoped receipts.
- Longer resident dense-SLM runs record memory and timing stability without fleet-wide performance claims.
- M4 BitNet proof command and receipt expectations are ready, but success remains blocked on an accepted BitNet artifact.

## Hard Constraints

- Do not execute MacBook artifact sweeps or MacBook receipts from this campaign.
- Do not reopen completed Apple M4 proof, operational, SLM answer, productization, performance, hardening, or dense regression campaigns.
- Do not weaken blocked BitNet local-answer gates.
- Do not claim BitNet local-answer quality from dense Qwen SLM evidence.
- Do not claim full `apple-m4-metal` inference, Neural Engine execution, MPSGraph model inference, QK256 support, or broad M4 performance.
- Do not touch QK256, `bitnet-qk256-dispatch`, server inference, or Metal kernels.
- Never commit model binaries.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-CONT-001 | merged | Added resident `bitnet mac chat` wrapper over the supported dense SLM warm-session runner. |
| M4-CONT-002 | pr_open | Add compact M4 dense-SLM golden smoke command. |
| M4-CONT-003 | proposed | Polish perceived dense-SLM latency from measured overhead. |
| M4-CONT-004 | proposed | Record longer resident-session memory and timing envelope. |
| M4-CONT-005 | blocked | Prepare M4 BitNet proof command and receipt contract after an accepted BitNet artifact exists. |

## Review Policy

Each PR owns one continuity item. Dense-SLM UX work must preserve `apple-m4-cpu-neon` routing, explicit fallback status, receipt validation, and the separation between dense Qwen evidence and BitNet proof. BitNet preparation must stay blocked until a reference-accepted artifact exists.
