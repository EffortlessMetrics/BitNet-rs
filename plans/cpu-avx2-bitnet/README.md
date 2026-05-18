# CPU AVX2 BitNet Hot-Path Plan

This plan owns the follow-on campaign that moves the official Microsoft BitNet
I2_S/QK256 CPU lane from correctness proof to production hot-path proof.

The governed end state is narrow: AVX2 CPU is considered fully working only
when the normal Rust CPU user path runs the official BitNet I2_S/QK256 artifact
with strict GGUF loader and tokenizer authority, intelligible answer receipts,
selected AVX2 BitNet kernels, no hidden scalar or dequant fallback, stable
scalar-vs-AVX2 token parity, and exact-profile phase timing good enough to
promote profile-by-profile.

## Authority

- Spec: `docs/specs/BITNET-SPEC-CPU-AVX2-HOTPATH.md`
- Existing CPU path authority: `docs/bitnet/BITNET_CPU_PATH_PLAN.md`
- Status page: `docs/bitnet/BITNET_CPU_AVX2_STATUS.md`
- Active tracker: `docs/tracking/campaigns/cpu-proof/active.toml`
- Durable proof-family boundary:
  `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`

## Scope

This lane is only for CPU AVX2 BitNet I2_S/QK256. It does not claim CUDA, NPU,
OpenVINO, A770, Apple M4, dense SLM, Qwen, server readiness, or broad chat
quality.

## Current next item

The first runtime item after this documentation item is
`CPU-AVX2-HOTPATH-002`: record QK256 hot-path execution counters and emit them
in strict CPU answer receipts. That diagnostic item must prove whether real
strict BitNet inference executes scaled I2_S x I8_S AVX2, scaled scalar, or the
older no-scale F32 QK256 GEMV path.
