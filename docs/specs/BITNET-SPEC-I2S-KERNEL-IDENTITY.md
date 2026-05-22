# BITNET-SPEC-I2S-KERNEL-IDENTITY

Status: active
Owner: BitNet-rs maintainers
Created: 2026-05-19
Linked proposal: docs/proposals/BITNET-PROP-0015-i2s-productization.md
Linked specs: self
Linked ADRs: n/a
Linked plan: plans/i2s/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: yes
Policy impact: no

Contract draft for I2_S lane. This spec captures route-specific acceptance, fallback-explicit receipt requirements, and non-inheritance boundaries.

## Hard rule

Production I2_S claims require explicit selected kernel identity, route
identity, and fallback state for strict accelerated proofs.

## Required kernel IDs

- qk256-scalar-f32-gemv
- qk256-scalar-f32-gemm
- qk256-scalar-i8s-scaled-gemv
- qk256-scalar-i8s-scaled-gemm
- qk256-avx2-f32-gemv
- qk256-avx2-i8s-scaled-gemv
- qk256-avx512-i8s-scaled-gemv
- qk256-neon-i8s-scaled-gemv
- qk256-cuda-i8s-scaled-gemv
- qk256-opencl-i8s-scaled-gemv
