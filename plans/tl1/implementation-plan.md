# TL1 Implementation Plan

Status: active
Owner: BitNet-rs contributors
Created: 2026-05-19
Linked proposal: docs/proposals/BITNET-PROP-0016-tl1-productization.md (planned)
Linked specs: docs/specs/BITNET-SPEC-TL1-ROUTE-CONTRACT.md (planned), docs/specs/BITNET-SPEC-TL1-LAYOUT.md (planned)
Linked ADRs: n/a
Linked plan: this document
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: planning only
Policy impact: none

## Objective

Make TL1 a first-class ARM table-lookup route with explicit route identity,
artifact authority, scalar-correctness proof, ARM NEON parity, Apple
CPU/NEON answer receipts, and exact-profile benchmark promotion.

## Work items

1. **TL1-PLAN-000 (docs source map and campaign setup)**
   - Add TL1 source map, plan lane, campaign manifest, and campaign summary.
   - Clarify ARM-first TL1 and x86 `unsupported_upstream` boundaries.
2. **TL1-SPEC-001 (proposal + route/layout specs)**
3. **TL1-SPEC-002 (scalar + ARM NEON specs)**
4. **TL1-SPEC-003 (artifact/model/quality specs)**
5. **TL1-SPEC-004 (Apple/performance/status specs)**
6. **TL1-LAYOUT-005 (layout reconciliation report)**
7. **TL1-FIXTURE-006 (synthetic TL1 fixture corpus)**
8. **TL1-SCALAR-007 (scalar oracle implementation)**
9. **TL1-SELECT-008 (selection/fallback metadata)**
10. **TL1-ARTIFACT-009+ (artifact/reference/loader/backend/benchmark/status PRs)**

## Proof commands

Run for docs/spec PRs in this lane:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check tl1
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

Runtime PRs may add narrower cargo test/bench commands per linked specs.

## Claim boundary

No TL1 runtime claim is valid until layout authority, scalar oracle,
artifact authority, and reference-good output are proven.
