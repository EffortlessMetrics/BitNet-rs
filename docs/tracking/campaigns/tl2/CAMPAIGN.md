# TL2 Campaign

Status: active
Owner: BitNet-rs contributors
Created: 2026-05-19
Linked proposal: docs/proposals/BITNET-PROP-0018-tl2-productization.md
Linked specs: docs/specs/BITNET-SPEC-TL2-ROUTE-CONTRACT.md, docs/specs/BITNET-SPEC-TL2-LAYOUT.md, docs/specs/BITNET-SPEC-TL2-SCALAR-ORACLE.md, docs/specs/BITNET-SPEC-TL2-X86-AVX.md, docs/specs/BITNET-SPEC-TL2-ARTIFACT-GATE.md, docs/specs/BITNET-SPEC-TL2-MODEL-COMPATIBILITY.md, docs/specs/BITNET-SPEC-TL2-REFERENCE-QUALITY.md, docs/specs/BITNET-SPEC-TL2-CPU.md, docs/specs/BITNET-SPEC-TL2-CUDA.md, docs/specs/BITNET-SPEC-TL2-PERFORMANCE.md, docs/specs/BITNET-SPEC-TL2-STATUS-SURFACE.md
Linked ADRs: docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md
Linked plan: plans/tl2/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: none

TL2 campaign scope is documentation/spec authority first, then layout/scalar/artifact/answer proof, and only then AVX/CUDA/performance promotion. Route receipts must be explicit: `selected_route=tl2`, `selected_kernel=tl2-*`, and `fallback_used=false` for strict proof runs.
