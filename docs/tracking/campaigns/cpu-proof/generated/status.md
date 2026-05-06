<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# BitNet CPU proof Campaign Status

- Campaign: `cpu-proof`
- State: `active`
- Objective: Make real BitNet CPU inference strict, receipt-backed, and measurable without routing around model, tokenizer, layout, or fallback truth.

## Work Items

| Item | State | PR | Branch | Acceptance |
|---|---|---:|---|---|
| CPU-BITNET-000 | merged | #3642 | `codex/cpu-proof/CPU-BITNET-000-path-plan` | Document the real BitNet CPU path implementation plan and sequence strict loader, tokenizer, layout, scalar, AVX2, receipts, and benchmarks. |
| CPU-BITNET-001 | merged | #3651 | `codex/cpu-proof/CPU-BITNET-001-loader-authority` | Strict CPU inference has one authoritative real GGUF loader path for BitNet models, and minimal fallback is impossible in strict proof mode. |
| CPU-BITNET-002 | merged | #3680 | `codex/cpu-proof/CPU-BITNET-002-tokenizer-authority` | Strict tokenizer resolution uses explicit override, GGUF metadata, sibling tokenizer assets, then strict failure. |
| CPU-BITNET-003 | merged | #3690 | `codex/cpu-bitnet-003-canonical-packed-layout` | Canonical block geometry, alignment, stride, and row/block iteration API are defined. |
| CPU-BITNET-004 | merged | #3696 | `codex/cpu-bitnet-004-scalar-packed-truth` | Canonical scalar packed QK256 GEMV/GEMM kernels are deterministic correctness oracles for decode and prefill. |
| CPU-BITNET-005 | ready | TBD | `codex/cpu-bitnet-005-avx2-gemv` | CPUID-gated AVX2/FMA QK256 GEMV dispatch matches the scalar packed oracle and records requested versus selected kernel identity. |

## Hard Constraints

- No GPU or NPU claims.
- No silent GGUF fallback.
- No performance claim without receipt artifacts.
- No helper-only SIMD work unless it is wired to real inference or explicitly scoped as preparation.
