# Qwen3.6 Implementation Plan

Status: active
Owner: BitNet-rs contributors
Created: 2026-05-19
Linked proposal: docs/proposals/BITNET-PROP-0017-qwen36-modern-dense-model-family.md
Linked specs: docs/specs/BITNET-SPEC-QWEN36-*
Linked ADRs: n/a
Linked plan: this file
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: candidate-only until proof receipts
Policy impact: strict claim boundaries

## Ordered phases

1. Source map + campaign registration + model-coverage registered rows.
2. Proposal and spec contracts for artifact/processor/architecture/reference/text-only/MoE/multimodal/memory/quality/performance/server/status.
3. Artifact inventory receipts (27B and 35B-A3B), no native claims.
4. External reference corpus receipts, no native claims.
5. Architecture and memory-envelope inventory.
6. Select one local quantized text-only candidate.
7. Native structural loader → CPU sanity/blocker → CUDA route ladder.
8. Exact-profile server/benchmark/promotion.

## Required proof commands for this planning PR

- `cargo run --locked -p xtask --no-default-features -- campaign check qwen36`
- `cargo run --locked -p xtask --no-default-features -- campaign generate --check`
- `cargo run --locked -p xtask --no-default-features -- check-model-coverage`
- `git diff --check`
