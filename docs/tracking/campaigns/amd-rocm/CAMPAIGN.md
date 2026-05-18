# AMD ROCm Productization Campaign

Campaign ID: `amd-rocm`

Status: active

## Objective

Register and sequence AMD ROCm as BitNet-rs's selected-device HIP/ROCm lane for
exact AMD Radeon, Radeon PRO, and Instinct routes without inheriting CUDA,
OpenCL, WGPU, CPU, OpenVINO, BitNet QK256, dense SLM, speed, residency, or
server proof.

## End State

- ROCm status surfaces answer which AMD GPU, ROCm/HIP runtime, GFX target,
  model family, route, fallback state, quality result, parity result, speed
  review, and not-claims apply.
- Selected backend labels are concrete AMD ROCm device routes, never generic
  `amd`, `gpu`, `rocm`, `hip`, or `radeon` labels.
- HIP source embedding, HIP compile, HIP runtime smoke, BitNet QK256 parity,
  dense SLM proof, answer quality, performance, residency, and server readiness
  remain separate proof stages.

## Hard Constraints

- Do not claim generic AMD GPU support.
- Do not claim ROCm from source text tests.
- Do not claim HIP compile from source embedding.
- Do not claim execution from compile smoke.
- Do not claim BitNet QK256 from dense SLM ROCm proof.
- Do not claim dense SLM from BitNet QK256 proof.
- Do not claim speedup without exact-profile review.
- Do not claim full residency without per-phase residency evidence.
- Do not add live ROCm hardware execution to ordinary PR CI.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| ROCM-DOCS-000 | ready | Add source-of-truth map, proposal, specs, plan, campaign tracker, specs index, and hardware matrix row. |
| ROCM-PROBE-001 | planned | Harden ROCm detection receipt and `bitnet rocm doctor --format json`. |
| ROCM-PROBE-002 | planned | Add strict selected-device identity. |
| ROCM-KERNEL-003 | planned | Add HIP compile-smoke harness. |
| ROCM-KERNEL-004 | planned | Add tiny HIP runtime smoke. |
| ROCM-RECEIPT-005 | planned | Validate ROCm receipt claim booleans. |
| ROCM-QK256-006..009 | planned | Lock QK256 fixtures, add HIP GEMV, persist context, and route strict QK256 through ROCm. |
| ROCM-QUALITY-010..012 | planned | Add BitNet one-token, corpus, long-decode, and warm-session proof. |
| ROCM-DENSE-013..016 | planned | Add dense SLM artifact, single-linear, decode, and warm-session proof. |
| ROCM-BENCH-017..018 | planned | Add profile timing receipts and exact-profile speed review. |
| ROCM-CLAIMS-019 | planned | Promote product CLI route only after proof gates. |
| ROCM-SERVER-020..021 | planned | Add exact-profile dense and BitNet server smoke. |

## Review Policy

ROCm PRs must keep device identity, runtime identity, route family, model
family, fallback truth, quality, parity, speed, residency, and server claims
separate. Documentation PRs may register the lane only; runtime PRs must link
the specific spec and plan item they implement.
