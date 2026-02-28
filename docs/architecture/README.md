# Architecture Documentation

This directory contains architecture documents describing the design of
BitNet-rs subsystems.

## Documents

| Document | Description |
|----------|-------------|
| [Multi-GPU Backend](multi-gpu-backend.md) | Multi-GPU backend design — Device enum, KernelProvider trait, KernelManager selection, CUDA/OpenCL coexistence, data flow, and future path |

## Architecture Decision Records

| ADR | Title |
|-----|-------|
| [ADR-001](decisions/ADR-001-production-model-baseline.md) | Production Model Baseline |
| [ADR-002](decisions/ADR-002-manual-branch-protection.md) | Manual Branch Protection |
| [ADR-003](decisions/ADR-003-receipt-schema-stability.md) | Receipt Schema Stability |
| [ADR-004](decisions/ADR-004-deterministic-baseline-tolerance.md) | Deterministic Baseline Tolerance |
# Architecture Decision Records
This directory contains Architecture Decision Records (ADRs) for the
BitNet-rs project.  Each ADR captures the context, decision, rationale,
and consequences of a significant architectural choice.
## Index
| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](decisions/ADR-001-production-model-baseline.md) | Production Model for CPU Baseline | Accepted | 2025-10-15 |
| [ADR-002](decisions/ADR-002-manual-branch-protection.md) | Manual Branch Protection | Accepted | 2025-10-15 |
| [ADR-003](decisions/ADR-003-receipt-schema-stability.md) | Receipt Schema Stability | Accepted | 2025-10-15 |
| [ADR-004](decisions/ADR-004-deterministic-baseline-tolerance.md) | Deterministic Baseline Tolerance | Accepted | 2025-10-15 |
| [ADR-005](decisions/ADR-005-opencl-first-backend.md) | OpenCL-First GPU Backend | Accepted | 2025-06-24 |
| [ADR-006](decisions/ADR-006-dynamic-gpu-linking.md) | Dynamic GPU Library Linking | Accepted | 2025-06-24 |
| [ADR-007](decisions/ADR-007-microcrate-per-backend.md) | Microcrate per GPU Backend | Accepted | 2025-06-24 |
| [ADR-008](decisions/ADR-008-kernel-embedding-strategy.md) | Kernel Embedding Strategy | Accepted | 2025-06-24 |
| [ADR-009](decisions/ADR-009-cpu-reference-testing.md) | CPU Reference Testing for GPU Kernels | Accepted | 2025-06-24 |
## Template
New ADRs should follow the structure in any of the existing records:
1. **Status** — Proposed → Accepted → Deprecated / Superseded
2. **Context** — What problem are we solving?
3. **Decision** — What did we decide?
4. **Rationale** — Why this option over others?
5. **Consequences** — Positive and negative impacts
6. **Alternatives Considered** — What else was evaluated?
