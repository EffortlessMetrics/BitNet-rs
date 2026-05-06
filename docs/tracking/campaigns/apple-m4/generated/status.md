<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 Mac mini validation Campaign Status

- Campaign: `apple-m4`
- State: `active`
- Objective: Make Apple Silicon a receipt-backed BitNet target through Metal, MPSGraph, and CPU/NEON proof lanes without conflating detection, execution, parity, inference, or performance.

## Work Items

| Item | State | PR | Branch | Acceptance |
|---|---|---:|---|---|
| M4-001 | merged | #3625 | `codex/hardware-scaffold/M4-001-apple-lane` | Add the Apple M4 Mac mini lane scaffold without runtime execution claims. |
| M4-002 | merged | #3627 | `codex/apple-m4/M4-002-profile-probe-bundle` | Add the Apple M4 Mac mini machine profile and probe bundle without runtime code, kernels, QK256, server inference, dependencies, or bulky artifacts. |
| M4-003 | merged | #3652 | `codex/apple-m4/M4-003-backend-identity` | Preserve Apple M4 Metal, MPSGraph, and CPU/NEON backend identity without adding runtime execution or kernels. |
| M4-004 | pr_open | #3692 | `codex/apple-m4/M4-004-metal-probe` | Add Apple M4 Metal device probe without claiming Metal execution or BitNet inference. |
| M4-005 | proposed | TBD | `codex/apple-m4/M4-005-metal-smoke` | Run a tiny native Metal compute smoke on M4 with fallback_used=false. |

## Hard Constraints

- Do not touch QK256 before a BitNet-specific Apple item explicitly allows it.
- Do not touch server inference.
- Do not claim Neural Engine execution from MPSGraph.
- Do not claim Metal execution from a Metal probe.
- Do not benchmark before parity exists.
