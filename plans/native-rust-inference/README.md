# Native Rust Inference Plan

This plan sequences the native Rust inference product lane described by
[BITNET-PROP-0003](../../docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md).
It turns the proof-carrying local inference direction into PR-sized work while
leaving active execution state in campaign `active.toml` files.

## Source-Of-Truth Links

| Surface | Path |
| --- | --- |
| Product proposal | `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md` |
| Model onboarding ladder | `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md` |
| Runtime performance contract | `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md` |
| Proof-family ADR | `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md` |
| CUDA product contract | `docs/specs/BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md` |
| Server readiness boundary | `docs/specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md` |
| Model claims | `ci/model-artifacts/model-coverage-matrix.toml` |
| NVIDIA campaign | `docs/tracking/campaigns/nvidia-5070ti/active.toml` |

## Plan Files

| File | Owns |
| --- | --- |
| `implementation-plan.md` | PR order and dependency map |
| `bitnet-official-i2s.md` | Official BitNet I2_S/QK256 performance and residency work |
| `bitnet-tl2.md` | TL/TL2 diagnostic lane boundaries |
| `dense-qwen25.md` | Dense Qwen2.5 optimization and requalification |
| `dense-qwen3.md` | Qwen3 product CLI promotion path |
| `smollm2.md` | SmolLM2 comparator and CPU/CUDA readiness path |
| `small-llm-candidates.md` | Llama/Gemma/Phi candidate order |
| `runtime-performance.md` | Timing, transfer, residency, and speedup proof work |
| `ci-economics.md` | Default PR cost and risk-routed expensive proof |
| `server-readiness.md` | Exact-profile server truth and receipt export |

## Operating Rules

- Do not create `.adze/goals`, `.bitnet/goals`, or another active work store.
- Use campaign manifests and events for active execution state.
- Keep generated dashboards generated.
- Keep BitNet, dense CUDA, benchmark, residency, and server proof families
  separate.
- Treat speedup, server readiness, and full residency as exact-profile claims
  until later specs promote broader scope.
