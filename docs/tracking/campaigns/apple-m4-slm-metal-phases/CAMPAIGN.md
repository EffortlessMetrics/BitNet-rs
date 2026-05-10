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
| M4-METAL-001 | merged | Selected prefill Q/K/V projection as the next phase target. |
| M4-METAL-002 | merged | Added the env-gated CPU/Metal Q/K/V parity fixture. |
| M4-METAL-003 | merged | Integrated phase receipt validation. |
| M4-METAL-004 | merged | Recorded the resident-route runtime boundary and prerequisite path. |
| M4-METAL-005 | merged | Promoted Q/K/V Metal dispatch from test-only fixture to runtime API. |
| M4-METAL-006 | in progress | Route the phase in resident sessions with parity. |
| M4-METAL-007 | blocked | Record measured phase-local timing deltas. |

## Current Decision

`M4-METAL-001` selected a prefill Q/K/V projection triplet as the next bounded
Metal target after the existing dense f32 prefill linear microphase. The
decision record is in `docs/slm/apple-m4-slm-metal-next-phase.md`.

`M4-METAL-002` added the parity fixture for that selected phase. The fixture is
env-gated for live Metal dispatch and remains outside resident generation.
`M4-METAL-003` promoted the phase evidence into validated receipt plumbing.
`M4-METAL-004` records the current runtime boundary: the live dispatch helper
is still test-local, so resident routing must wait for a non-dev Metal runtime
API. `M4-METAL-005` owns that runtime extraction before resident routing.

## Claim Boundary

Metal phase evidence does not prove full `apple-m4-metal` inference, Neural
Engine execution, MPSGraph model inference, QK256 support, BitNet quality, or
broad Apple Silicon performance.
