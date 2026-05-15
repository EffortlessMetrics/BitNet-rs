<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 inference ops Campaign Status

- Campaign: `apple-m4-inference-ops`
- State: `active`
- Objective: Turn the completed Apple M4 dense SLM and BitNet proof surfaces into a durable operator layer: status, report inventory, advisory refresh, regression dashboarding, disk/cache posture, and an operator envelope v2 with explicit claim boundaries.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-INF-OPS-001 | merged | #4960 | `codex/apple-m4-inference-ops/M4-INF-OPS-001-mac-status` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add `bitnet mac status` with an `apple_m4_inference_status` receipt covering dense SLM, BitNet, disk/cache, report inventory, commands, and explicit claim boundaries without running live inference. |
| M4-INF-OPS-002 | merged | #4963 | `codex/apple-m4-inference-ops/M4-INF-OPS-002-report-refresh` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add advisory/nightly report refresh manifest generation for committed M4 dense SLM and BitNet report families without running live models in generic PR CI. |
| M4-INF-OPS-003 | proposed | TBD | `codex/apple-m4-inference-ops/M4-INF-OPS-003-regression-dashboard` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add compact regression dashboard artifacts across dense SLM and BitNet reports while keeping evidence families, model identity, tokenizer authority, backend, fallback, and claim boundaries separate. |
| M4-INF-OPS-004 | proposed | TBD | `codex/apple-m4-inference-ops/M4-INF-OPS-004-operator-envelope-v2` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Publish operator envelope v2 mapping supported M4 commands to receipt requirements, gates, report families, and unsupported claim boundaries. |

## Hard Constraints

- This is an M4 Mac mini operations campaign.
- Do not reopen completed Apple M4 dense SLM, local server, Metal phase, BitNet local-answer, BitNet eval/benchmark, or BitNet productization campaigns unless a regression proves they are wrong.
- Do not use dense Qwen evidence as BitNet evidence.
- Do not enable BitNet chat or BitNet serve in this campaign.
- Do not claim full apple-m4-metal inference, QK256 support, Neural Engine execution, MPSGraph model inference, MacBook evidence, broad Apple Silicon performance, or speedup.
- Do not add live model downloads, hardware timing runs, or long resident soaks to generic required PR CI.
- Never commit model binaries.
