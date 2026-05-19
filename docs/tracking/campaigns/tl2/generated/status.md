<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# TL2 x86 table lookup route Campaign Status

- Campaign: `tl2`
- State: `active`
- Objective: Govern TL2 as a distinct x86-first table-lookup proof family with source-of-truth docs, draft specs, and campaign scaffolding before any runtime, model, or benchmark claim.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| TL2-DOCS-000 | ready | TBD | `codex/define-tl2-as-x86-table-lookup-route` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Register the TL2 x86 table-lookup route as documentation and tracker scaffolding only. The slice must add the source map, proposal, draft spec set, implementation plan, campaign tracker entry, and status/index links without claiming native runtime support, model compatibility, answer quality, performance, CPU/CUDA proof, server support, GPU/NPU execution, TL1 proof inheritance, or BitNet QK256/I2_S changes. |

## Hard Constraints

- TL2 registration is not native BitNet-rs inference support.
- Do not edit runtime code, kernels, model binaries, server inference, GPU/NPU execution, or BitNet QK256/I2_S kernels.
- Do not claim answer quality, performance, CPU/CUDA parity, or model compatibility until follow-up receipts prove them.
