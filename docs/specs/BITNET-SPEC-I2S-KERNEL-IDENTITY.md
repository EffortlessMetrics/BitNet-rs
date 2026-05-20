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

Production I2_S claims require explicit selected kernel, selected route, selected backend, fallback status, and proof-family booleans for strict accelerated proofs.

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

Receipt explainers may display `qk256_gemv_cuda` as a user-facing kernel
summary for the production CUDA GEMV path, but receipts must retain the
selected kernel ID and backend-specific counters needed to distinguish
production packed/scaled I2_S/QK256 kernels from diagnostic F32/no-scale
QK256 probes.
