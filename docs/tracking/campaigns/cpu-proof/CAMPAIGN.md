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
| CPU-BITNET-001 | merged | Strict GGUF loader authority merged in #3651. |
| CPU-BITNET-002 | merged | Strict tokenizer authority merged in #3680. |
| CPU-BITNET-003 | merged | Canonical packed layout merged in #3690. |
| CPU-BITNET-004 | merged | Scalar truth kernels merged in #3696. |
| CPU-BITNET-005a | merged | AVX2/FMA feature plumbing merged in #3735. |
| CPU-BITNET-005b | merged | Requested/selected QK256 kernel selection merged in #3748. |
| CPU-BITNET-005c | merged | AVX2 decode GEMV parity hardening merged in #3753. |
| CPU-BITNET-006 | merged | CPU transformer decode step merged in #3793. |
| CPU-BITNET-007 | merged | Strict CPU proof receipt enforcement merged in #3800. |
| CPU-BITNET-008 | merged | CPU phase benchmark receipt profiles merged in #3856. |
| CPU-PHASE-TIMING-001 | merged | Tightened Kaby phase timing receipt extraction in #3872 while leaving micro/layer gaps explicit. |
| CPU-ANSWER-001 | merged | Strict CPU answer-readiness gates merged in #3898. |
| CPU-ANSWER-002 | merged | Scalar-vs-AVX2 full-decode answer parity merged in #3906, with the 258V CPU as the lead BitNet CPU reference machine. Runs against a rejected answer artifact are diagnostic-only until `MODEL-ARTIFACT-002` provides an `answer_ready` artifact. |
| CPU-ANSWER-003 | ready | Reference-divergence triage is next: compare strict BitNet CPU answer runs against known-good reference artifacts before blaming AVX2 or claiming answer quality. |

## Review Policy

CPU proof PRs are non-stackable when they touch loader, tokenizer, layout, dispatch, decode, or receipt authority. New BitNet CPU proof leadership routes through the 258V CPU lane; other CPU machines are support validators unless a work item explicitly says otherwise. Hardware validation lanes can consume CPU proof artifacts but should not rewrite the same runtime surface concurrently.
