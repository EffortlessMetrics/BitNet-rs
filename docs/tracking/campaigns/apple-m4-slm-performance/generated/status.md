<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 SLM performance Campaign Status

- Campaign: `apple-m4-slm-performance`
- State: `active`
- Objective: Turn the working Apple M4 SLM answer path into a fast, efficient, repeatable local-answer engine by measuring release-mode warm-session bottlenecks, removing hot-loop overhead, improving CPU/NEON execution, expanding only parity-gated Metal phases, and keeping every speed claim tied to receipts.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-SLM-PERF-001 | merged | #4044 | `codex/apple-m4-slm-performance/M4-SLM-PERF-001-release-baseline` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a release-mode Apple M4 SLM warm-session baseline for warm_16, warm_32, warm_64, and warm_128 profiles, separating cold load from warm prompt timing and recording model/tokenizer/backend/fallback/session fields without making broad performance claims. |
| M4-SLM-PERF-002 | pr_open | #4047 | `codex/apple-m4-slm-performance/M4-SLM-PERF-002-allocation-audit` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Audit decode-loop allocations, logits and sampling scratch allocation, token vector growth, detokenization/string churn, receipt construction, and temporary tensor creation; name unavoidable per-token allocations before any math optimization work. |
| M4-SLM-PERF-003 | proposed | TBD | `codex/apple-m4-slm-performance/M4-SLM-PERF-003-resident-session` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Make resident model/tokenizer reuse the normal multi-prompt path, with session-owned buffers, safe runtime-state reuse, per-prompt receipts, and an aggregate receipt separating model_load, tokenize, prefill, decode, sampling, and total timing. |
| M4-SLM-PERF-004 | proposed | TBD | `codex/apple-m4-slm-performance/M4-SLM-PERF-004-cpu-neon-hotpath` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Optimize the highest measured Apple M4 CPU/NEON bottleneck while preserving greedy token IDs, deterministic quality corpus results, fallback status, and before/after warm-session receipts. |
| M4-SLM-PERF-005 | proposed | TBD | `codex/apple-m4-slm-performance/M4-SLM-PERF-005-metal-phase-expansion` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Expand only named Apple Metal prefill/projection phases after CPU baseline stability, requiring CPU-only versus CPU-plus-Metal greedy parity, Metal phase fallback_used=false, explicit CPU fallback for the rest of the pipeline, and timing delta receipts. |
| M4-SLM-PERF-006 | proposed | TBD | `codex/apple-m4-slm-performance/M4-SLM-PERF-006-streaming-ux` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add streaming token output, time-to-first-token receipts, quiet default logs, operator-friendly progress, and clear failure messages without changing backend claim boundaries. |
| M4-SLM-PERF-007 | proposed | TBD | `codex/apple-m4-slm-performance/M4-SLM-PERF-007-performance-envelope` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Publish a measured Apple M4 SLM performance envelope for supported models and profiles only, recording machine context, backend, profile, timings, phase contributions, fallback status, and explicit unsupported claims. |

## Hard Constraints

- Do not reopen the completed apple-m4, apple-m4-operational, apple-m4-slm-answer, or apple-m4-productization campaigns.
- Do not weaken the blocked BitNet apple-m4-local-answer gates.
- Do not claim BitNet local-answer quality from dense SLM evidence.
- Do not claim full apple-m4-metal inference from named Metal phases.
- Do not claim Neural Engine execution, MPSGraph model inference, QK256 support, or broad M4 performance.
- Do not optimize cold start before release-mode warm-session bottlenecks are measured.
- Never commit model binaries.
- Prefer the verified local model cache and keep downloads storage-conscious.
