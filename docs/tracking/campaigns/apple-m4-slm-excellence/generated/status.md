<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 SLM excellence Campaign Status

- Campaign: `apple-m4-slm-excellence`
- State: `active`
- Objective: Turn the working Apple M4 dense SLM path into an appliance-grade local model runner experience with native-feeling CLI, reliable health checks, lower perceived latency, better allocation hygiene, stronger quality coverage, longer resident-session stability, leading dense SLM support, local regression reporting, a measured operator envelope, and efficient default CI.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-SLM-EX-001 | merged | #4276 | `codex/apple-m4-slm-excellence/M4-SLM-EX-001-mac-doctor` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add bitnet mac doctor as a one-command local health verdict for the supported dense Apple M4 SLM path, checking cache presence, model hash, disk headroom, tiny ask/smoke behavior, receipt validation, backend/fallback identity, and unsupported-backend rejection without downloading models by default. |
| M4-SLM-EX-002 | merged | #4280 | `codex/apple-m4-slm-excellence/M4-SLM-EX-002-chat-polish` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Polish bitnet mac chat as a resident local tool with clean prompt loop behavior, EOF/Ctrl-C handling, quiet default logs, streaming by default, optional per-turn receipts, aggregate session receipt at exit, and clear model/tokenizer loaded-once status. |
| M4-SLM-EX-003 | merged | #4284 | `codex/apple-m4-slm-excellence/M4-SLM-EX-003-ttft` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Measure and reduce dense Mac SLM time-to-first-token overhead in prompt template construction, tokenization, first decode step, streaming flush behavior, cache verification placement, or receipt construction while preserving greedy token IDs and quality corpus behavior. |
| M4-SLM-EX-004 | merged | #4286 | `codex/apple-m4-slm-excellence/M4-SLM-EX-004-allocation-cleanup` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Clean resident decode hot-loop allocations for sampling scratch, logits buffers where supported, token vector growth, detokenization string churn, temporary tensors, and receipt JSON placement while preserving output quality, timing, and receipt schema. |
| M4-SLM-EX-005 | merged | #4288 | `codex/apple-m4-slm-excellence/M4-SLM-EX-005-model-matrix` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Define an M4 dense SLM model support matrix for the default model and leading candidate SLMs, with default, supported, candidate, and rejected states, recording source, file, size, SHA256, tokenizer authority, prompt template, quantization, Rust support status, M4 support status, cache policy, and quality status without accepting unsupported models. |
| M4-SLM-EX-006 | merged | #4293 | `codex/apple-m4-slm-excellence/M4-SLM-EX-006-second-model` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a second storage-conscious dense instruct model only if it passes reference output sanity, Rust M4 output quality, tokenizer authority checks, cache metadata verification, and receipt validation. |
| M4-SLM-EX-007 | merged | #4297 | `codex/apple-m4-slm-excellence/M4-SLM-EX-007-quality-corpus-v2` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Expand to a small quality corpus 2.0 covering factual answers, instruction following, format-constrained output, one-sentence generation, arithmetic, small summarization, and short rewrite while preserving fast local runtime and validated receipts. |
| M4-SLM-EX-008 | merged | #4301 | `codex/apple-m4-slm-excellence/M4-SLM-EX-008-long-soak` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Record 25-prompt and 50-prompt resident dense SLM sessions with 64-token and 128-token response budgets, memory drift, time-to-first-token drift, decode throughput drift, quality failures, and model/tokenizer reuse. |
| M4-SLM-EX-009 | merged | #4305 | `codex/apple-m4-slm-excellence/M4-SLM-EX-009-local-regression` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add bitnet mac regression as a local advisory comparison command against matching stored M4 dense SLM envelope receipts, with optional hard-fail mode for model/tokenizer/backend/fallback/quality/timing/memory drift, while keeping live model and hardware performance checks out of generic required CI. |
| M4-SLM-EX-010 | in_progress | TBD | `codex/apple-m4-slm-excellence/M4-SLM-EX-010-user-envelope` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Publish a measured M4 mini user expectation envelope covering cold load, warm ask timing, time-to-first-token, warm 16/32/64/128 timing, decode tokens per second, peak memory, cache size, and known unsupported models/backends. |

## Hard Constraints

- This is an M4 Mac mini local campaign.
- Do not execute MacBook artifact sweeps or MacBook receipts here.
- Do not reopen completed Apple M4 proof, operational, SLM answer, productization, performance, hardening, continuity, or dense regression campaigns.
- Do not weaken blocked BitNet local-answer gates.
- Do not claim BitNet local-answer quality from dense Qwen SLM evidence.
- Do not claim full apple-m4-metal inference, Neural Engine execution, MPSGraph model inference, QK256 support, or broad M4 performance.
- Do not touch QK256, bitnet-qk256-dispatch, server inference, or Metal kernels unless a later phase-scoped Metal item explicitly allows it.
- Do not add live model downloads, long resident soaks, or hardware performance runs to generic required CI; keep those as local, advisory, or scheduled Apple-hardware checks.
- Never commit model binaries.
