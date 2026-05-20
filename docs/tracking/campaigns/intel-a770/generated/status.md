<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Intel Arc A770 validation Campaign Status

- Campaign: `intel-a770`
- State: `active`
- Objective: Validate Intel Arc A770 as an OpenCL-first BitNet acceleration lane with selected-device receipts and no CPU, NPU, CUDA, OpenVINO GPU, or uncommitted proof conflation.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| A770-000 | merged | #5623 | `codex/intel-a770/A770-000-truth-reconciliation` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Reconcile active.toml, CAMPAIGN.md, route matrix, kernel matrix, claim ledger, and model contract so no full inference or trusted-partial A770 claim is present without committed claim-grade receipts. |
| A770-003 | merged | #5969 | `codex/intel-a770/A770-003-backend-identity` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Preserve Intel Arc A770 requested and selected backend identity without adding kernels or inference claims. |
| A770-004 | merged | #5971 | `codex/intel-a770/A770-004-runtime-probe` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add A770 runtime probe evidence without promoting OpenCL execution or BitNet inference claims. |
| A770-005 | merged | #6072 | `codex/intel-a770/A770-005-opencl-smoke` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run a tiny selected-device OpenCL smoke with fallback_used=false and CPU parity, without BitNet inference claims. |
| A770-006 | merged | #6110 | `codex/intel-a770/A770-006-opencl-parity` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add CPU/OpenCL parity for a minimal kernel or subgraph without promoting official BitNet QK256 inference. |
| A770-007 | merged | #6113 | `codex/intel-a770/A770-007-receipt-identity` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Record selected-device receipt identity for the validated smoke/parity path before any performance or trusted-partial claim. |

## Hard Constraints

- OpenCL-first for native A770 proof.
- Uncommitted transcript evidence cannot promote A770 claims.
- OpenVINO GPU is reference only.
- Generic OpenCL and Arc 140V proof cannot count as A770 proof.
- CPU fallback cannot count as A770 execution.
