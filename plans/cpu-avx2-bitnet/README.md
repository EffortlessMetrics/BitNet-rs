# CPU AVX2 BitNet Hot-Path Plan

Status: active
Owner: Codex
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-CPU-AVX2-HOTPATH.md`, `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: n/a
Linked plan: `plans/cpu-avx2-bitnet/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Exact-profile CPU AVX2 BitNet support only after receipts prove it.
Policy impact: none

## Purpose

This plan moves the BitNet CPU lane from correctness proof to production hot-path
proof for the official Microsoft BitNet I2_S/QK256 artifact. The immediate
campaign question is narrow: does strict real Rust CPU inference execute the
scaled I2_S x I8_S QK256 AVX2 path, or does it only carry an AVX2 label while
running scalar, dequantized, diagnostic, or no-scale F32-style work?

The target end state is a normal Rust CPU user path with strict GGUF loader and
tokenizer authority, correct answer-corpus behavior, selected AVX2 BitNet
kernels, no hidden scalar/dequant fallback, scalar-versus-AVX2 parity, and phase
receipts good enough to promote exact profiles one by one.

## Source-of-truth links

| Surface | Path |
| --- | --- |
| Repo source-of-truth rules | `docs/reference/SPEC_SYSTEM.md` |
| Hot-path behavior spec | `docs/specs/BITNET-SPEC-CPU-AVX2-HOTPATH.md` |
| CPU path plan | `docs/bitnet/BITNET_CPU_PATH_PLAN.md` |
| Kernel matrix | `docs/bitnet/BITNET_KERNEL_MATRIX.md` |
| Receipt fields | `docs/bitnet/BITNET_RECEIPT_FIELDS.md` |
| Runtime performance contract | `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md` |
| Campaign active goal | `docs/tracking/campaigns/cpu-proof/active.toml` |
| Hot-path implementation plan | `plans/cpu-avx2-bitnet/implementation-plan.md` |

## Operating rules

- Scalar packed QK256 remains the correctness oracle for every AVX2 kernel.
- Strict requested AVX2 must fail closed if AVX2/FMA or the exact selected AVX2
  kernel is unavailable.
- Receipts must record requested and selected backend, requested and selected
  kernel, kernel family, runtime API, fallback status, model identity, loader
  mode, quant format, hash, tokenizer source, and strict tokenizer status.
- Production BitNet inline-scale inference must not substitute the no-scale F32
  AVX2 GEMV for the scaled I2_S x I8_S path.
- Performance claims require phase receipts and profile-by-profile review.
- This lane does not claim CUDA, NPU, OpenVINO, server readiness, dense SLM
  readiness, Qwen readiness, or broad chat quality.

## Plan files

| File | Owns |
| --- | --- |
| `README.md` | Scope, source links, and operating rules |
| `implementation-plan.md` | PR order, acceptance, proof commands, and rollback |

## First follow-on runtime question

The first runtime PR after these docs must add hot-path counters and receipts
that distinguish all of the following paths:

```text
qk256_f32_scalar_gemv_invocations
qk256_f32_avx2_gemv_invocations
qk256_i8s_scaled_scalar_invocations
qk256_i8s_scaled_avx2_invocations
qk256_flat_bytes_extracted_count
qk256_input_rows_materialized_count
qk256_output_rows_allocated_count
qk256_tensor_to_vec_count
```

No optimization PR should start until those counters can prove which path strict
real BitNet inference actually executes.
