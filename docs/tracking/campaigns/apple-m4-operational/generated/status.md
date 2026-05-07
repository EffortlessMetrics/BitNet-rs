<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 operational readiness Campaign Status

- Campaign: `apple-m4-operational`
- State: `active`
- Objective: Turn the completed Apple M4 proof lane into a repeatable operator workflow with one-command validation, durable receipts, clear docs, stable failure modes, and benchmark profiles.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-OP-001 | merged | #3845 | `codex/apple-m4-operational/M4-OP-001-validation-bundle` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a one-command Apple M4 validation bundle that runs the known-good proof sequence, writes the expected receipts under a caller-provided output directory, validates every receipt, and writes a summary that states exactly what is and is not proven. |
| M4-OP-002 | merged | #3848 | `codex/apple-m4-operational/M4-OP-002-receipt-bundle-validator` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Harden the Apple M4 receipt-bundle validator so it fails on missing fallback fields, backend/fallback mismatches, CPU fallback counted as Metal proof, MPSGraph-as-Neural-Engine claims, missing BitNet fields, or premature QK256 Apple claims. |
| M4-OP-003 | merged | #3857 | `codex/apple-m4-operational/M4-OP-003-operator-runbook` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add an Apple M4 operator runbook that documents model download and placement, validation commands, expected receipts, backend label meanings, failure modes, and unsupported claims. |
| M4-OP-004 | pr_open | #3871 | `codex/apple-m4-operational/M4-OP-004-effective-cli-flow` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Document and, where needed, polish effective-use CLI examples for strict Apple CPU/NEON BitNet proof and receipt-backed Metal phase proof, with clear strict failure-mode messaging. |
| M4-OP-005 | merged | #3861 | `codex/apple-m4-operational/M4-OP-005-benchmark-profile-summary` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add conservative Apple M4 benchmark profile names, expected timing fields, and summary artifact validation without turning tiny-kernel timings into broad performance claims. |
| M4-OP-006 | proposed | TBD | `codex/apple-m4-operational/M4-OP-006-next-frontier` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Decide and document the next Apple implementation frontier: CPU/NEON usability, Metal subgraph expansion, or QK256 investigation, with claim boundaries and a new campaign if implementation work should continue. |

## Hard Constraints

- Do not reopen the completed apple-m4 proof campaign.
- Do not claim full apple-m4-metal model inference unless a strict real-model receipt proves it.
- Do not claim Neural Engine execution from MPSGraph.
- Do not claim QK256 on Apple Silicon.
- Do not claim general M4 performance from tiny-kernel benchmarks.
