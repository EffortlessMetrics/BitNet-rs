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

## Plan intent

Treat TL2 as a distinct x86-first table-lookup proof family. The implementation order is layout-safe first, scalar-correct second, artifact-authoritative third, answer-good fourth, and optimized fifth.

## Hard rails

- TL2 is x86-first and route-distinct from I2_S/QK256 and TL1.
- ARM TL2 remains `unsupported_upstream` for tracked model families unless the compatibility ledger changes with upstream evidence.
- TL2 cannot inherit I2_S/QK256, TL1, or dense SLM answer/backend/speed proof.
- No AVX/CUDA promotion before scalar TL2 oracle proof.
- No answer claim before TL2 artifact gate + reference-good outputs.
- No speedup claim before exact-profile benchmark review.

## Phase order

1. **Source-of-truth docs and tracker scaffolding** (source map, campaign, plan, matrix/index links).
2. **Proposal/spec authority** (route contract, layout, scalar, AVX, artifact, compatibility, quality, CPU/CUDA/performance/status).
3. **Layout reconciliation** (`quantization-support.md`, `tl2.rs`, `tl_lut.rs`, and upstream/reference evidence).
4. **Synthetic TL2 fixtures** (pack/unpack correctness, route rejection negatives, metadata validation).
5. **Scalar TL2 oracle and kernel-selection metadata** (strict fail-closed + fallback-explicit behavior).
6. **Artifact and reference authority** (official 2B and 3B runner/conversion receipts; reference-quality corpus).
7. **Loader and strict CPU proof** (selected_route/selected_kernel/fallback receipts).
8. **AVX2 and AVX-512 parity proof** (scalar-token parity, tail parity, deterministic behavior).
9. **Optional CUDA TL2 candidate** (only after x86 CPU answer-ready proof).
10. **Benchmarks and promotion review** (exact-profile only; no blanket speed claims).

## Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check tl2
cargo run --locked -p xtask --no-default-features -- campaign generate --check
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```
