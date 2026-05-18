# Architectural Decision Records

ADRs record durable BitNet-rs decisions. Use them when a lane needs a stable
architecture, proof, or policy choice that should survive individual PRs and
still matter months later.

ADRs do not own active work state, PR queues, or generated dashboards. Active
execution state belongs in `.bitnet/goals/active.toml` or the selected
campaign-local `active.toml`; generated dashboards are derived from campaign
manifests, events, and receipts.

## Current ADRs

- ADR-0001: [Configuration layering and clamp location](./0001-configuration-layering.md)
- ADR-0002: [GPU Backend Strategy](./0002-gpu-backend-strategy.md)
- BITNET-ADR-0004: [9950X3D + RTX 5070 Ti CUDA Product Bench](./BITNET-ADR-0004-9950x3d-5070ti-cuda-product-bench.md)

## Source-of-truth role

| Layer | Owns |
| --- | --- |
| Proposal | Why the effort exists |
| Spec | What must be true |
| ADR | What decision was made and why it is durable |
| Plan | PR order and proof commands |
| Active goal or campaign manifest | Current executable work |
| Policy TOML | Enforceable ledger |
| Receipt or artifact | Evidence |

For BitNet proof work, ADRs should keep claim boundaries explicit. For example,
an ADR may decide that answer-ready model artifacts must precede backend answer
claims, or that dense SLM proof is first-class but must not be treated as
BitNet I2_S or QK256 proof.

## Template for new ADRs

Copy as `docs/adr/NNNN-title.md` or use a BitNet-prefixed filename when a
cross-lane proof decision needs a stable identifier:

```md
# ADR-NNNN: Title

Status: Proposed | Accepted | Superseded by NNNN | Rejected
Date: YYYY-MM-DD
Owner:
Linked proposal:
Linked specs:
Linked plan:

## Decision

## Context

## Consequences

## Alternatives considered

## Follow-up specs / plans

## Claim boundary

## How to revert
```
