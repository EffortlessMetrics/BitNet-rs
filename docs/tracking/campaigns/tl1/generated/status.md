<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# TL1 ARM table lookup route Campaign Status

- Campaign: `tl1`
- State: `active`
- Objective: Govern TL1 as an ARM-first table-lookup proof family with source-of-truth docs, an implementation plan, and compatibility-ledger boundaries before any runtime, kernel, artifact, or answer-quality claim.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| TL1-PLAN-000 | ready | TBD | `codex/add-tl1-as-arm-first-route` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Register the TL1 ARM-first table-lookup route as documentation and tracker scaffolding only. The slice must add the source map, implementation plan, campaign tracker entry, compatibility-ledger references, and status/index links without claiming native runtime support, model compatibility, answer quality, performance, CPU/Metal proof, server support, GPU/NPU execution, TL2 proof inheritance, or BitNet QK256/I2_S changes. |

## Hard Constraints

- TL1 registration is not native BitNet-rs inference support.
- Do not edit runtime code, kernels, model binaries, server inference, GPU/NPU execution, or BitNet QK256/I2_S kernels.
- Do not claim answer quality, performance, CPU/Metal proof, or model compatibility until follow-up receipts prove them.
