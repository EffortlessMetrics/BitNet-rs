# TL1 Source Map

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

## Purpose

This source map establishes TL1 as an ARM-first BitNet table-lookup route with
its own route identity, artifact authority, layout proof, scalar oracle,
backend proof, answer quality receipts, and benchmark promotion ladder.

## Route boundary

- TL1 is a separate route family from `I2_S`/QK256.
- TL1 is a separate route family from TL2.
- For currently tracked families, x86 TL1 remains
  `unsupported_upstream` in the compatibility ledger.
- ARM TL1 support listings are candidate inputs, not proof of answer readiness.

## Authority stack for TL1 lane

1. Proposal: `docs/proposals/BITNET-PROP-0016-tl1-productization.md`.
2. Specs: `docs/specs/BITNET-SPEC-TL1-*.md`.
3. Plan: `plans/tl1/implementation-plan.md`.
4. Active campaign goal: `docs/tracking/campaigns/tl1/active.toml`.
5. Compatibility ledger: `ci/model-artifacts/model-kernel-compatibility.toml`.

## Current claim boundary

TL1 remains planning/specification-only in this PR. No answer/backend/speed
promotion is introduced by this source map.
