# CPU Proof Campaign

Campaign ID: `cpu-proof`

Status: active

## Objective

Make real BitNet CPU inference strict, receipt-backed, and measurable. The campaign must prove loader truth, tokenizer truth, canonical packed layout, scalar correctness, SIMD dispatch, transformer decode coverage, strict fallback behavior, and benchmarks in that order.

## End State

- Real GGUF loading is authoritative in strict proof mode.
- Tokenizer resolution is strict and receipt-backed.
- Packed BitNet layout is canonical.
- Scalar truth kernels exist before SIMD performance work.
- AVX2 dispatch and decode work are parity checked.
- Strict receipts record model, tokenizer, quantization, kernel family, selected backend, phase, and fallback status.
- Benchmarks are tied to hardware context and receipt artifacts.

## Hard Constraints

- Do not route around real BitNet loading or tokenizer authority.
- Do not claim CPU performance before strict receipts and benchmark artifacts exist.
- Do not mix GPU, NPU, or server inference work into CPU proof items.
- Do not treat helper-only AVX2 code as real inference.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| CPU-BITNET-000 | merged | Real BitNet CPU path plan merged in #3642. |
| CPU-BITNET-001 | pr_open | Strict GGUF loader authority is open in #3651. |
| CPU-BITNET-002 | proposed | Strict tokenizer authority. |
| CPU-BITNET-003 | proposed | Canonical packed layout. |
| CPU-BITNET-004 | proposed | Scalar truth kernels. |
| CPU-BITNET-005 | proposed | AVX2 decode GEMV. |
| CPU-BITNET-006 | proposed | CPU transformer decode ops. |
| CPU-BITNET-007 | proposed | Strict receipts and fallback behavior. |
| CPU-BITNET-008 | proposed | CPU phase benchmark profiles. |

## Review Policy

CPU proof PRs are non-stackable when they touch loader, tokenizer, layout, dispatch, decode, or receipt authority. Hardware validation lanes can consume CPU proof artifacts but should not rewrite the same runtime surface concurrently.
