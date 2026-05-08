<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 local answer productization Campaign Status

- Campaign: `apple-m4-productization`
- State: `active`
- Objective: Turn the completed Apple M4 SLM proof lane into a practical Mac user flow with documented baseline commands, model cache management, Mac-oriented CLI wrappers, warm-session speed polish, and a parity-gated Metal phase handoff.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-PROD-001 | merged | #4017 | `codex/apple-m4-productization/M4-PROD-001-user-facing-baseline` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Document the working Rust-native Apple M4 CPU/NEON SLM local-answer baseline, including the current warm-session command, expected model artifact, receipt fields, failure boundaries, and unsupported claims. |
| M4-PROD-002 | pr_open | #4020 | `codex/apple-m4-productization/M4-PROD-002-model-cache` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add storage-conscious model cache commands to fetch, verify, list, and prune supported SLM artifacts under the user cache, recording source, size, SHA256, and tokenizer metadata while warning on low disk and never committing binaries. |
| M4-PROD-003 | proposed | TBD | `codex/apple-m4-productization/M4-PROD-003-mac-cli` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add Mac-oriented check, ask, validate, and receipts-check CLI wrappers that route the supported SLM path through apple-m4-cpu-neon with strict loader/tokenizer behavior and clear errors for missing models, wrong hashes, unsupported backends, hidden fallback, and premature full Metal requests. |
| M4-PROD-004 | proposed | TBD | `codex/apple-m4-productization/M4-PROD-004-warm-speed-polish` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Polish warm-session timing and operator thresholds so 16, 32, and 64 token warm answers are measured with cold load separated, model/tokenizer reuse visible, and no broad performance claim. |
| M4-PROD-005 | proposed | TBD | `codex/apple-m4-productization/M4-PROD-005-metal-phase` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Implement the first Apple Metal prefill linear projection microphase only with CPU-only versus CPU-plus-Metal greedy parity, Metal phase fallback_used=false, the rest of the pipeline recorded as CPU/NEON, layout handling recorded, and no full Metal inference claim. |

## Hard Constraints

- Do not reopen the completed apple-m4, apple-m4-operational, or apple-m4-slm-answer campaigns.
- Do not weaken the blocked BitNet apple-m4-local-answer gates.
- Do not claim BitNet local-answer quality from dense SLM evidence.
- Do not claim full apple-m4-metal inference until a strict real-model receipt proves it.
- Do not claim Neural Engine execution from MPSGraph or any unresolved Apple graph target.
- Do not claim QK256 support on Apple Silicon from SLM evidence.
- Do not claim broad performance from warm-session or tiny phase receipts.
- Never commit model binaries.
