# Architectural Decision Records

| ADR | Title | Status |
|-----|-------|--------|
| [0001](./0001-configuration-layering.md) | Configuration layering and clamp location | Accepted |
| [0002](./0002-gpu-backend-strategy.md) | GPU Backend Strategy | Accepted |
| [001](./ADR-001-opencl-initial-backend.md) | Use OpenCL as initial Intel GPU backend | Accepted |
| [002](./ADR-002-microcrate-per-backend.md) | One microcrate per GPU backend | Accepted |
| [003](./ADR-003-gpu-hal-abstraction.md) | GPU Hardware Abstraction Layer (HAL) traits | Accepted |
| [004](./ADR-004-kernel-compilation-strategy.md) | Runtime kernel compilation with caching | Accepted |
| [005](./ADR-005-cpu-fallback-strategy.md) | Automatic CPU fallback when GPU unavailable | Accepted |
| [006](./ADR-006-feature-flag-design.md) | Feature flag structure for multi-backend | Accepted |

## Template for new ADRs

Copy as `docs/adr/NNNN-title.md`:

```md
# ADR-NNNN: Title
- **Status:** Proposed | Accepted | Superseded by NNNN | Rejected
- **Date:** YYYY-MM-DD
- **Context:** <problem / forces>
- **Decision:** <what we chose and why>
- **Consequences:** <positive/negative trade-offs>
- **Alternatives considered:** <brief bullets>
- **How to revert:** <what to change back if needed>
```
