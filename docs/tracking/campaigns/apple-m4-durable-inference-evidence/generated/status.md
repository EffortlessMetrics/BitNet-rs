<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 durable inference evidence Campaign Status

- Campaign: `apple-m4-durable-inference-evidence`
- State: `active`
- Objective: Turn the completed Apple M4 dense SLM and BitNet proof surfaces into durable, repeatable evidence: longer resident dense SLM benchmark profiles, refreshed matching-identity report pairs, dashboard comparisons with real history, and operator envelopes that describe measured drift without broad Apple Silicon or model-quality claims.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-DURABLE-001 | merged | #4973 | `codex/apple-m4-durable-inference-evidence/M4-DURABLE-001-resident-100-profile` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add `resident_100` to the dense Apple M4 SLM benchmark v2 profile contract, CLI parser, receipt validator allowlist, tests, and operator docs without claiming that a fresh 100-prompt live M4 run has been recorded. |
| M4-DURABLE-002 | merged | #4988 | `codex/apple-m4-durable-inference-evidence/M4-DURABLE-002-dense-refresh` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run and commit a fresh dense SLM eval/benchmark refresh for every supported M4 model identity, including `resident_100`, with receipt validation and regression comparison against the previous matching reports. |
| M4-DURABLE-003 | proposed | TBD | `codex/apple-m4-durable-inference-evidence/M4-DURABLE-003-bitnet-refresh` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run and commit a fresh BitNet eval, benchmark, and variable warm-session refresh for the accepted artifact/tokenizer identity with receipt validation and matching-identity regression comparison. |
| M4-DURABLE-004 | proposed | TBD | `codex/apple-m4-durable-inference-evidence/M4-DURABLE-004-dashboard-history` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Regenerate M4 report-refresh and regression-dashboard artifacts so dense SLM and BitNet families have comparable matching-history groups with explicit thresholds, warnings, or failures. |
| M4-DURABLE-005 | proposed | TBD | `codex/apple-m4-durable-inference-evidence/M4-DURABLE-005-operator-envelope-v3` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Publish an operator envelope refresh describing the durable M4 refresh cadence, regression thresholds, resident-100 status, disk/cache guidance, and claim boundaries from matching-history reports. |

## Hard Constraints

- This is an M4 Mac mini evidence-refresh campaign.
- Do not reopen completed Apple M4 dense SLM eval v2, BitNet eval/benchmark, BitNet productization, inference-ops, local server, or Metal campaigns unless a regression proves they are wrong.
- Do not use dense Qwen evidence as BitNet evidence.
- Do not enable BitNet chat or BitNet serve in this campaign.
- Do not claim full apple-m4-metal inference, QK256 support, Neural Engine execution, MPSGraph model inference, MacBook evidence, broad Apple Silicon performance, broad model quality, or speedup.
- Do not add live model downloads, hardware timing runs, BitNet runtime runs, or long resident soaks to generic required PR CI.
- Never commit model binaries.
