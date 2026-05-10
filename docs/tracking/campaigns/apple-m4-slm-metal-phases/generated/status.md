<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 SLM Metal phases Campaign Status

- Campaign: `apple-m4-slm-metal-phases`
- State: `active`
- Objective: Expand Apple M4 dense SLM acceleration through named, phase-scoped Metal prefill/projection contributions with CPU-only versus CPU-plus-Metal parity, Metal fallback_used=false receipts, explicit CPU/NEON routing for the rest of the pipeline, and timing deltas, without claiming full apple-m4-metal inference.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-METAL-001 | merged | #4376 | `codex/apple-m4-slm-metal-phases/M4-METAL-001-next-phase` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Choose the next Apple Metal dense SLM phase candidate, recording why it is safe, CPU reference scope, expected tensor shape, parity method, timing fields, fallback rules, and claim boundaries without adding kernels or routing. |
| M4-METAL-002 | merged | #4379 | `codex/apple-m4-slm-metal-phases/M4-METAL-002-parity-fixture` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add the CPU reference versus CPU-plus-Metal parity fixture for the selected phase, requiring same greedy-relevant outputs or a bounded documented tolerance, and no hidden fallback. |
| M4-METAL-003 | merged | #4382 | `codex/apple-m4-slm-metal-phases/M4-METAL-003-phase-receipt` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Integrate a phase receipt for the selected Metal contribution with selected_backend=apple-m4-metal, runtime_api=metal, fallback_used=false, timing delta, CPU remainder routing, and mac receipts-check validation. |
| M4-METAL-004 | merged | #4385 | `codex/apple-m4-slm-metal-phases/M4-METAL-004-runtime-boundary` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Record the resident-routing feasibility boundary for the validated Q/K/V Metal phase, explicitly documenting that live dispatch is currently test-local, choosing the required runtime extraction path, and blocking resident routing until a non-dev Metal runtime API exists. |
| M4-METAL-005 | in_progress | TBD | `codex/apple-m4-slm-metal-phases/M4-METAL-005-runtime-api` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Promote the validated dense prefill Q/K/V Metal dispatch from test-only code into a non-dev runtime API with target/feature-gated dependencies, generic CI-safe fixture tests, live M4 opt-in proof, and no resident generation routing. |
| M4-METAL-006 | blocked | TBD | `codex/apple-m4-slm-metal-phases/M4-METAL-006-resident-route` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Route the runtime Q/K/V Metal phase into resident-session flow only where parity holds, with CPU-only versus CPU-plus-Metal greedy comparison, quality corpus pass, explicit CPU/NEON routing for non-Metal phases, and phase receipts per relevant turn. |
| M4-METAL-007 | blocked | TBD | `codex/apple-m4-slm-metal-phases/M4-METAL-007-measured-delta` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Record measured phase-local timing deltas for the resident-routed Metal phase, keeping speedup claims phase-local and updating docs without broad M4 or full Metal inference claims. |

## Hard Constraints

- This is an M4 Mac mini dense SLM campaign.
- Do not claim full apple-m4-metal inference from a named phase.
- Do not claim Apple Metal accelerates the full SLM answer path.
- Do not claim BitNet local-answer quality, QK256 support, Neural Engine execution, MPSGraph model inference, or broad M4 performance.
- Do not touch QK256, bitnet-qk256-dispatch, server inference, or MacBook work.
- Never commit model binaries.
