# TL1 Campaign

Status: active
Owner: BitNet-rs contributors
Created: 2026-05-19
Linked proposal: docs/proposals/BITNET-PROP-0016-tl1-productization.md (planned)
Linked specs: docs/specs/BITNET-SPEC-TL1-ROUTE-CONTRACT.md (planned)
Linked ADRs: n/a
Linked plan: plans/tl1/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: planning only
Policy impact: none

## Campaign thesis

TL1 is an ARM-first BitNet table-lookup route that must be productized as its
own lane instead of inheriting proof from `I2_S`/QK256 or TL2.

## Current work item

- `TL1-PLAN-000`: docs/source-of-truth bootstrap only.
- No runtime, answer, backend, or speed promotion in this item.

## Route boundaries

- TL1 is not `I2_S`/QK256.
- TL1 is not TL2.
- x86 TL1 remains `unsupported_upstream` for tracked model families unless the
  upstream support table changes and the compatibility ledger is updated.
- ARM TL1 support listings require per-artifact, per-route, and per-backend
  proof before claims.

## Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check tl1
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```
