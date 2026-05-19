# TL2 Implementation Plan

Status: active
Owner: BitNet-rs contributors
Created: 2026-05-19
Linked proposal: docs/proposals/BITNET-PROP-0018-tl2-productization.md
Linked specs: docs/specs/BITNET-SPEC-TL2-ROUTE-CONTRACT.md, docs/specs/BITNET-SPEC-TL2-LAYOUT.md, docs/specs/BITNET-SPEC-TL2-SCALAR-ORACLE.md, docs/specs/BITNET-SPEC-TL2-X86-AVX.md, docs/specs/BITNET-SPEC-TL2-ARTIFACT-GATE.md, docs/specs/BITNET-SPEC-TL2-MODEL-COMPATIBILITY.md, docs/specs/BITNET-SPEC-TL2-REFERENCE-QUALITY.md, docs/specs/BITNET-SPEC-TL2-CPU.md, docs/specs/BITNET-SPEC-TL2-CUDA.md, docs/specs/BITNET-SPEC-TL2-PERFORMANCE.md, docs/specs/BITNET-SPEC-TL2-STATUS-SURFACE.md
Linked ADRs: docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md
Linked plan: self
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none until proof gates pass
Policy impact: none

## Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check tl2
cargo run --locked -p xtask --no-default-features -- campaign generate --check
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```
