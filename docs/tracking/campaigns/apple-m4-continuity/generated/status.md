<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 continuity Campaign Status

- Campaign: `apple-m4-continuity`
- State: `active`
- Objective: Keep the M4 Mac mini useful as the local Apple Silicon dense-SLM appliance while BitNet artifact qualification happens elsewhere, by improving resident Mac UX, smoke validation, latency polish, and blocked BitNet proof preparation without widening hardware claims.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-CONT-001 | in_progress | TBD | `codex/apple-m4-continuity/M4-CONT-001-mac-chat` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a resident `bitnet mac chat` wrapper for the supported dense Qwen Apple M4 CPU/NEON path, accepting repeated prompts or stdin, streaming by default, reusing the existing warm-session runner so the model/tokenizer load once, writing aggregate and per-prompt receipts, and preserving device-boundary errors before cache/model work. |
| M4-CONT-002 | proposed | TBD | `codex/apple-m4-continuity/M4-CONT-002-golden-smoke` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a compact M4 dense-SLM golden smoke command that verifies cache health, asks a tiny prompt, checks receipt validity, records backend/fallback identity, and reports disk/cache health without creating new model or performance claims. |
| M4-CONT-003 | proposed | TBD | `codex/apple-m4-continuity/M4-CONT-003-latency-polish` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Polish perceived latency for the dense Mac SLM path by improving streaming flush behavior, quiet default logs, prompt formatting overhead, sampling/logits scratch reuse, or receipt construction placement based on measured receipts while preserving greedy token IDs and quality corpus behavior. |
| M4-CONT-004 | proposed | TBD | `codex/apple-m4-continuity/M4-CONT-004-resident-envelope` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Record a longer dense-SLM resident-session memory and timing envelope for repeated 64-token and 128-token prompts, proving model/tokenizer reuse and receipt stability without changing model support or broadening performance claims. |
| M4-CONT-005 | blocked | TBD | `codex/apple-m4-continuity/M4-CONT-005-bitnet-proof-prep` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Prepare the M4-side strict BitNet CPU/NEON proof command shape, required receipt schema, and accepted-artifact input contract so the M4 proof can run once an artifact is accepted, while failing clearly when the artifact is missing or not accepted. |

## Hard Constraints

- This is an M4 Mac mini local campaign; do not execute MacBook artifact sweeps or MacBook receipts here.
- Do not reopen completed Apple M4 proof, operational, SLM answer, productization, performance, hardening, or dense regression campaigns.
- Do not weaken blocked BitNet local-answer gates.
- Do not claim BitNet local-answer quality from dense Qwen SLM evidence.
- Do not claim full apple-m4-metal inference, Neural Engine execution, MPSGraph model inference, QK256 support, or broad M4 performance.
- Do not touch QK256, bitnet-qk256-dispatch, server inference, or Metal kernels.
- Never commit model binaries.
