<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Intel Arc A770 validation Campaign Status

- Campaign: `intel-a770`
- State: `active`
- Objective: Validate Intel Arc A770 as an OpenCL-first BitNet acceleration lane with selected-device receipts and no CPU, NPU, CUDA, or OpenVINO GPU conflation.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| A770-000 | ready | TBD | `codex/intel-a770/A770-000-truth-reconciliation` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Reconcile the A770 campaign, model contract, route matrix, kernel matrix, claims ledger, and committed receipts so they all preserve diagnostic current state and no full-inference claim appears without claim-grade evidence. |
| A770-003 | proposed | TBD | `codex/intel-a770/A770-003-backend-identity` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Preserve Intel Arc A770 requested and selected backend identity without adding kernels or inference claims. |

## Hard Constraints

- OpenCL-first for native A770 proof.
- OpenVINO GPU is reference only.
- CPU fallback cannot count as A770 execution.
- Diagnostic A770 route rows and model-contract targets do not promote to trusted partial until claim-grade receipts are committed.
