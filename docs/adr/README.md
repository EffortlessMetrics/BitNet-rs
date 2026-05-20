# Architectural Decision Records

ADRs record durable BitNet-rs decisions. Use them when a lane needs a stable
architecture, proof, or policy choice that should survive individual PRs.

ADRs do not own active work state or generated dashboards. Active execution
state belongs in campaign-local `active.toml`; generated dashboards are derived
from campaign manifests and events.

## Current ADRs

- ADR-0001: [Configuration layering and clamp location](./0001-configuration-layering.md)
- ADR-0002: [GPU Backend Strategy](./0002-gpu-backend-strategy.md)
- BITNET-ADR-0004: [9950X3D + RTX 5070 Ti CUDA Product Bench](./BITNET-ADR-0004-9950x3d-5070ti-cuda-product-bench.md)
- BITNET-ADR-0005: [Proof Families Are Not Interchangeable](./BITNET-ADR-0005-proof-families-are-not-interchangeable.md)

## Source-Of-Truth Role

| Layer | Owns |
| --- | --- |
| Proposal | Why the effort exists |
| Spec | What must be true |
| ADR | What decision was made and why it is durable |
| Plan | PR order and proof commands |
| Campaign `active.toml` | Current executable work |
| Handoff | Operator transfer context and closeout notes |
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
- **Status:** Proposed | Accepted | Superseded by NNNN | Rejected
- **Date:** YYYY-MM-DD
- **Linked proposal/spec:** <paths>
- **Context:** <problem / forces>
- **Decision:** <what we chose and why>
- **Consequences:** <positive/negative trade-offs>
- **Claim boundary:** <what this decision does not prove>
- **Alternatives considered:** <brief bullets>
- **How to revert:** <what to change back if needed>
```
