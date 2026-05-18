# CPU AVX2 BitNet hot-path plan

Status: Draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: ../../docs/specs/BITNET-SPEC-CPU-AVX2-HOTPATH.md
Linked ADRs: ../../docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md
Linked plan: implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: CPU AVX2 BitNet I2_S/QK256 remains candidate/profile-scoped until hot-path and phase receipts promote it.
Policy impact: none

## Goal

Move the CPU AVX2 BitNet lane from correctness proof to production hot-path
proof for the official Microsoft I2_S/QK256 artifact. The immediate objective is
not generic AVX2 optimization; it is proving which QK256 path strict real-model
inference actually executes, then implementing and wiring the scaled I2_S x I8_S
AVX2 path if it is missing.

## Documents

- [Spec](../../docs/specs/BITNET-SPEC-CPU-AVX2-HOTPATH.md) defines strict
  fallback, receipt, parity, hot-path, and claim rules.
- [Implementation plan](implementation-plan.md) sequences PR-sized work from
  counters through profile-specific support status.
- [Status](../../docs/bitnet/BITNET_CPU_AVX2_STATUS.md) records the current
  claim boundary and near-term board.

## Current claim boundary

The current repository may claim strict CPU correctness and answer-corpus proof
only to the extent already backed by existing CPU proof receipts. It must not
claim production-grade AVX2 BitNet hot-path performance until receipts show the
scaled I2_S x I8_S AVX2 path was selected, invoked, did not fall back, preserved
scalar parity, and met exact-profile timing gates.
