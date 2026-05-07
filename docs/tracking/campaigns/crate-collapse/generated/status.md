<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Crate collapse Campaign Status

- Campaign: `crate-collapse`
- State: `active`
- Objective: Collapse low-risk public microcrates into SRP modules while preserving behavior, feature gates, and public API intent.

## Work Items

| Item | State | PR | Branch | Acceptance |
|---|---|---:|---|---|
| INV-001 | merged | #3632 | `codex/inventory/INV-001-crate-consolidation-map` | Add crate consolidation inventory mapping every workspace member to a final public crate or internal module without moving code. |
| LEAF-001 | proposed | TBD | `codex/crate-collapse/LEAF-001-warn-once` | Move warn-once code under bitnet-common, update imports and workspace membership, and preserve behavior. |

## Hard Constraints

- Do not combine crate movement with runtime proof.
- Do not change hardware lane semantics.
- Do not move domain crates before leaf dependencies are settled.
- Do not mass-format unrelated files.
