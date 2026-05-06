# CPU QK256 Performance Campaign

Campaign ID: `cpu-qk256-performance`

Status: active

## Objective

Move CPU BitNet work from strict proof surfaces into scalar, packed-layout, AVX2, and sustained benchmark evidence without hiding fallback behavior.

## End State

- Scalar truth kernels are available before SIMD dispatch claims.
- QK256 and related packed layouts are receipt-backed.
- AVX2 paths can be forced and compared against scalar references.
- Mobile and desktop baselines record thermal, power, memory, and sustained-load context.

## Hard Constraints

- Do not claim performance before strict proof receipts exist.
- Do not mix GPU, NPU, CUDA, OpenCL, or Metal execution into CPU proof.
- Do not treat helper-only AVX2 code as real inference.
- Do not report short turbo behavior as sustained performance.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| KBL8250U-003 | merged | Prove i5-8250U scalar and AVX2 dispatch. |
| KBL8250U-004 | ready | Add strict CPU proof run on the i5-8250U lane. |

## Review Policy

CPU QK256 and performance PRs are non-stackable when they touch dispatch, packed layouts, receipts, or benchmark interpretation.
