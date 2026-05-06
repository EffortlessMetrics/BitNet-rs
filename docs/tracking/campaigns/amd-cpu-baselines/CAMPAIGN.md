# AMD CPU Baselines Campaign

Campaign ID: `amd-cpu-baselines`

Status: active

## Objective

Validate AMD 5700X and 9950X3D as CPU proof and benchmark lanes while preserving scalar, AVX2, AVX-512, cache, memory, and sustained-power context.

## End State

- 5700X proves mainstream Zen 3 AVX2 behavior and never claims AVX-512.
- 9950X3D proves Zen 5 AVX2 and AVX-512 behavior with cache-domain context.
- Benchmark artifacts separate correctness proof from performance claims.

## Hard Constraints

- These lanes are CPU proof lanes, not accelerator lanes.
- The 5700X lane must not claim AVX-512.
- 9950X3D receipts must record scheduler/core placement and cache-domain context before performance claims.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| AMD5700X-003 | ready | Prove 5700X scalar and AVX2 dispatch. |
| AMD9950X3D-003 | ready | Prove 9950X3D scalar, AVX2, and AVX-512 dispatch. |

## Review Policy

AMD CPU baseline PRs should keep 5700X and 9950X3D claims separate unless an item explicitly owns a shared comparison artifact.
