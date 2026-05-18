# Falcon3 Family Campaign

Status: active
Owner: model-artifacts
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0012](../../../proposals/BITNET-PROP-0012-falcon3-family-supported-models.md)
Linked specs: [Falcon3 specs](../../../specs/INDEX.md#falcon3-family-onboarding)
Linked ADRs: [BITNET-ADR-0005](../../../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [Falcon3 family implementation plan](../../../../plans/falcon3-family/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: candidate registration only until receipts promote exact rows
Policy impact: no policy exception

## Scope

This campaign registers Falcon3 as BitNet-rs's first multi-size BitNet-family onboarding lane. It starts with documentation, artifact authority, tokenizer/prompt authority, route compatibility, and reference-quality contracts. Runtime, backend, performance, and server claims require later receipts.

## Current Item

`F3-000` adds source-of-truth docs, spec contracts, and candidate-only matrix rows. It must not add model binaries, runtime changes, answer-readiness claims, backend-readiness claims, speedup claims, or server-readiness claims.

## Proof Commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check falcon3-family
cargo run --locked -p xtask --no-default-features -- campaign generate --check
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```
