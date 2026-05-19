# I2_S Implementation Plan

Status: active
Owner: BitNet-rs maintainers
Created: 2026-05-19
Linked proposal: docs/proposals/BITNET-PROP-0015-i2s-productization.md
Linked specs: docs/specs/BITNET-SPEC-I2S-*.md
Linked ADRs: n/a
Linked plan: self
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: yes
Policy impact: no

## Goal

Make I2_S/QK256 a governed, explicit production lane with canonical layout, scaled-kernel identity, fallback-explicit receipts, and conservative model-family claim control.

## PR order

1. Source-map and campaign scaffolding.
2. Proposal/spec contracts.
3. Layout authority cleanup in layout-core + quantization alignment.
4. Scaled kernel ID split and selection metadata.
5. CPU hot-path cleanup and scaled prefill GEMM.
6. AVX2/AVX512/NEON scaled parity lanes.
7. CUDA/OpenCL receipt alignment.
8. Model-family artifact-gated expansion.
9. Performance and status surfaces.

## Proof commands (docs/spec PRs)

- `cargo run --locked -p xtask --no-default-features -- campaign check i2s`
- `cargo run --locked -p xtask --no-default-features -- campaign generate --check`
- `git diff --check`
