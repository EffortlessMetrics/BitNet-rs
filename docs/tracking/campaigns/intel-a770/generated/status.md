<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Intel Arc A770 validation Campaign Status

- Campaign: `intel-a770`
- State: `active`
- Objective: Validate Intel Arc A770 as an OpenCL-first BitNet acceleration lane with selected-device receipts and no CPU, NPU, CUDA, or OpenVINO GPU conflation.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| A770-003 | ready | TBD | `codex/intel-a770/A770-003-backend-identity` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Preserve Intel Arc A770 requested and selected backend identity without adding kernels or inference claims. |

## Hard Constraints

- OpenCL-first for native A770 proof.
- OpenVINO GPU is reference only.
- CPU fallback cannot count as A770 execution.
