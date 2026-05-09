# Apple M4 SLM Metal Phases

Campaign ID: `apple-m4-slm-metal-phases`

Status: active

## Objective

Expand Apple M4 dense SLM acceleration phase by phase while preserving CPU/NEON
as the honest full-pipeline route until a later strict full-route receipt proves
otherwise.

## End State

- Each Metal phase has a CPU reference.
- CPU-only and CPU-plus-Metal outputs pass parity for the phase scope.
- Metal receipts record `fallback_used=false`, timing deltas, and CPU routing
  for remaining phases.
- Resident-session routing is added only after parity and receipt validation.
- No full `apple-m4-metal` inference claim is made from phase evidence.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-METAL-001 | ready | Choose the next phase target and receipt shape. |
| M4-METAL-002 | pending | Add CPU/Metal parity fixture. |
| M4-METAL-003 | pending | Integrate phase receipt validation. |
| M4-METAL-004 | pending | Route the phase in resident sessions with parity. |
| M4-METAL-005 | pending | Record measured phase-local timing deltas. |

## Claim Boundary

Metal phase evidence does not prove full `apple-m4-metal` inference, Neural
Engine execution, MPSGraph model inference, QK256 support, BitNet quality, or
broad Apple Silicon performance.
