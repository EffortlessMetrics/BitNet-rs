# Apple M4 SLM Performance Campaign

Campaign ID: `apple-m4-slm-performance`

Status: active

## Objective

Turn the working Apple M4 SLM answer path into a fast, efficient, repeatable local-answer engine by measuring release-mode warm-session bottlenecks, removing hot-loop overhead, improving CPU/NEON execution, expanding only parity-gated Metal phases, and keeping every speed claim tied to receipts.

## Why This Exists

The `apple-m4-slm-answer` and `apple-m4-productization` campaigns proved the practical Mac baseline: a sub-1 GiB dense instruct GGUF can answer through the Rust CLI on `apple-m4-cpu-neon`, with model cache management, Mac CLI wrappers, warm-session receipts, deterministic quality checks, and a first named Metal phase handoff.

That is working, but not yet excellent. This campaign owns the next layer: release-mode baselines, warm-session overhead, resident-session hardening, CPU/NEON speed work, measured Metal phase expansion, and user-facing streaming/latency polish.

## End State

- Release-mode warm-session baselines exist for `warm_16`, `warm_32`, `warm_64`, and `warm_128` with cold load separated.
- Hot-loop allocation and temporary-buffer churn are audited and bounded before math optimization.
- Resident sessions are the normal path for multi-prompt local use, with per-prompt and aggregate receipts.
- CPU/NEON optimizations preserve greedy token IDs and quality receipts.
- Metal phases expand only behind CPU parity, `fallback_used=false`, phase timing, and explicit CPU fallback for the rest of the pipeline.
- User-facing performance envelopes are published only after receipts back the named profile, model, backend, and machine context.

## Hard Constraints

- Do not reopen the completed `apple-m4`, `apple-m4-operational`, `apple-m4-slm-answer`, or `apple-m4-productization` campaigns.
- Do not weaken the blocked BitNet `apple-m4-local-answer` gates.
- Do not claim BitNet local-answer quality from dense SLM evidence.
- Do not claim full `apple-m4-metal` inference from a named Metal phase.
- Do not claim Neural Engine execution, MPSGraph model inference, QK256 support, or broad M4 performance.
- Do not optimize cold start before warm-session bottlenecks are measured.
- Do not commit model binaries.
- Keep downloads storage-conscious and prefer the existing verified model cache.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-SLM-PERF-001 | merged | Add release-mode warm-session baseline profiles for 16, 32, 64, and 128 generated tokens. |
| M4-SLM-PERF-002 | merged | Audit decode and receipt hot-loop allocations before optimizing math. |
| M4-SLM-PERF-003 | merged | Harden resident-session reuse as the normal multi-prompt path. |
| M4-SLM-PERF-004 | in_progress | Optimize the measured CPU/NEON bottleneck while preserving greedy output. |
| M4-SLM-PERF-005 | proposed | Expand only parity-gated Metal prefill/projection phases with explicit fallback boundaries. |
| M4-SLM-PERF-006 | proposed | Add streaming, time-to-first-token receipts, quiet logs, and operator-friendly progress. |
| M4-SLM-PERF-007 | proposed | Publish a measured performance envelope for supported profiles and hardware only. |

## Review Policy

Each PR owns one work item. Performance PRs must include a receipt-backed claim boundary: which model, profile, backend, phase, machine context, and fallback status were measured. Speedups require before/after receipts with the same prompt/profile settings. Metal work must remain phase-scoped unless a later strict full-inference receipt proves otherwise.
