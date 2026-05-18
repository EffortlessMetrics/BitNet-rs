# BITNET-SPEC-CUDA-ROUTE-CONTRACT: CUDA Route Contract

Status: proposed
Owner: inference/product
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
Linked specs: [BITNET-SPEC-0013](BITNET-SPEC-0013-model-onboarding-proof-ladder.md), [BITNET-SPEC-0014](BITNET-SPEC-0014-runtime-performance-contract.md), [BITNET-SPEC-0007](BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md), [BITNET-SPEC-0010](BITNET-SPEC-0010-server-readiness-proof-boundary.md)
Linked ADRs: [BITNET-ADR-0004](../adr/BITNET-ADR-0004-9950x3d-5070ti-cuda-product-bench.md)
Linked plan: [CUDA 5070 Ti productization](../../plans/cuda-5070ti-productization/README.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines route and proof-family fields; does not promote any model row.
Policy impact: n/a

## Purpose

CUDA claims in BitNet-rs must identify the model family, backend, route, proof
family, execution plan, and fallback result that actually ran. This contract
adds route-level vocabulary under the existing model onboarding and runtime
performance specs so dense CUDA, official BitNet QK256 CUDA, server shared
engine CUDA, CPU reference, speedup, and full-residency claims cannot be
mistaken for one another.

This spec is docs-only. It does not implement new routing, receipts, kernels,
CLI behavior, server behavior, benchmarks, model downloads, or model coverage
promotion.

## Source-Of-Truth Authorities

This spec relies on:

- [Native Rust inference product proposal](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
- [Model onboarding proof ladder](BITNET-SPEC-0013-model-onboarding-proof-ladder.md)
- [Runtime performance contract](BITNET-SPEC-0014-runtime-performance-contract.md)
- [9950X3D + RTX 5070 Ti CUDA product contract](BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md)
- [Server readiness proof boundary](BITNET-SPEC-0010-server-readiness-proof-boundary.md)
- [9950X3D + RTX 5070 Ti CUDA product bench ADR](../adr/BITNET-ADR-0004-9950x3d-5070ti-cuda-product-bench.md)
- [CUDA capability matrix](../status/CUDA_CAPABILITY_MATRIX.md)
- `ci/model-artifacts/model-coverage-matrix.toml`
- `ci/hardware/windows-9950x3d-rtx5070ti/**`

Receipts remain the evidence for what happened. This spec defines the minimum
route fields that future CUDA receipts, status surfaces, and validators must
preserve before they promote or explain a CUDA proof claim.

## Required CUDA Route IDs

CUDA receipts and status summaries must use stable route identifiers when they
claim CUDA execution or route readiness:

| Route ID | Scope | May prove | Must not prove |
| --- | --- | --- | --- |
| `bitnet_qk256_cuda` | Official BitNet I2_S/QK256 packed CUDA route | Exact official BitNet packed I2_S/QK256 CUDA execution for the named artifact/backend/profile | Dense regular-LLM CUDA, TL1/TL2, GPU-int2, global speedup, full residency, broad server readiness |
| `dense_regular_llm_cuda` | Dense SLM or small dense LLM CUDA route | Exact dense model-family CUDA execution for the named artifact/backend/profile | BitNet packed I2_S/QK256, BitNet 1-bit/TL proof, other dense model rows |
| `dense_gguf_linear_cuda_parity` | Dense GGUF linear/kernel parity fixture route | Kernel or layer fixture parity evidence for the scoped dense GGUF artifact | Full dense inference, server readiness, product CLI readiness, BitNet proof |
| `dense_gguf_layer_plan` | Dense GGUF planning and gap-audit route | Execution-plan readiness or unsupported-op accounting | CUDA execution, answer quality, speedup, full residency |
| `server_shared_engine_cuda` | CUDA server shared-engine route | Exact server endpoint/profile evidence when paired with model route, request, response, and receipt fields | Broad production serving, streaming, concurrency, speedup, full residency |

A later spec may add route IDs, but it must state the model family, proof
family, allowed claims, forbidden claims, and receipt fields for the new route.

## Required Fields In Every CUDA Proof Receipt

Every receipt that supports a CUDA execution, CUDA readiness, or CUDA route
claim must include or normalize to these fields:

```json
{
  "requested_backend": "cuda | nvidia-rtx-5070-ti-cuda",
  "selected_backend": "nvidia-rtx-5070-ti-cuda",
  "runtime_api": "cuda",
  "selected_route": "bitnet_qk256_cuda | dense_regular_llm_cuda | dense_gguf_linear_cuda_parity | dense_gguf_layer_plan | server_shared_engine_cuda",
  "fallback_used": false,
  "fallback_reason": null,
  "execution_plan": {
    "route": "...",
    "bitnet_qk256_cuda_ops": 0,
    "dense_regular_llm_cuda_ops": 0,
    "cpu_fallback_ops": 0,
    "unsupported_ops": 0
  },
  "proof_family": {
    "bitnet_packed_i2s_qk256_proof": true,
    "dense_regular_llm_cuda_proof": false
  }
}
```

Receipts may include additional route-specific fields, but these fields must not
be omitted from the normalized receipt explanation for CUDA claims. If an
execution plan is not applicable because the receipt is only a planner or gap
audit, the receipt must say that explicitly and must not promote a CUDA
execution claim.

## Backend Resolution Rules

- A user convenience selector such as `cuda` may be accepted by CLI or server
  commands, but a proof claim must resolve it to `nvidia-rtx-5070-ti-cuda`
  before reporting CUDA proof on the 9950X3D + RTX 5070 Ti lane.
- `requested_backend="cuda"` without a strict `selected_backend` is not RTX
  5070 Ti proof.
- `selected_backend="cuda"` is not strict proof. The selected backend must name
  the concrete backend identity before route or model coverage promotion.
- CPU AVX-512, WGPU, Vulkan, OpenCL, D3D12, hardware visibility, CUDA driver
  detection, and NVML probes are useful context, but they are not CUDA route
  execution proof without selected-route evidence and fallback rejection.

## Proof-Family Boundaries

Proof families are non-interchangeable:

- Dense CUDA can never satisfy official BitNet packed I2_S/QK256 proof.
- Official BitNet packed I2_S/QK256 CUDA can never satisfy dense SLM or small
  dense LLM proof.
- Dense GGUF linear parity can never satisfy full dense inference proof without
  model-level route, tokenizer, prompt, decode, and answer-quality receipts.
- Dense GGUF layer plans can never satisfy CUDA execution proof; they may only
  describe route eligibility and unsupported operations.
- Server shared-engine CUDA proof must identify both the server route and the
  underlying model route before it can support an exact-profile server claim.
- CPU reference proof can support answer sanity or comparator evidence, but it
  cannot prove CUDA execution, CUDA speedup, or CUDA residency.

## Fallback Rules

Strict CUDA receipts must fail closed:

- `fallback_used=true` is a hard failure for strict CUDA execution proof.
- `cpu_fallback_ops > 0` is a hard failure for strict CUDA execution proof.
- A missing `fallback_used` field must be treated as unknown, not as false.
- A missing `fallback_reason` is acceptable only when `fallback_used=false` is
  explicit.
- Fallback-free kernel execution does not by itself prove answer quality,
  product CLI readiness, speedup, server readiness, or full residency.

## Execution Plan Rules

A CUDA receipt with no execution plan cannot promote a CUDA claim. The plan must
make the route and operation counts auditable enough to reject proof-family
conflation:

- `bitnet_qk256_cuda` receipts must record positive BitNet QK256 CUDA work for
  the scoped proof and zero CPU fallback operations.
- `dense_regular_llm_cuda` receipts must record positive dense CUDA work for the
  scoped proof and zero CPU fallback operations.
- Planner receipts may record unsupported operations, but unsupported-op
  accounting is not execution.
- Server receipts must carry request/response identity plus the model route that
  generated the response.

## Claim Promotion Rules

CUDA route receipts may support only the claim family they prove:

| Claim | Required CUDA route evidence | Additional gate |
| --- | --- | --- |
| BitNet QK256 CUDA proof | `bitnet_qk256_cuda` execution plan and fallback rejection | Official BitNet artifact, tokenizer, prompt policy, answer-quality receipt |
| Dense regular-LLM CUDA proof | `dense_regular_llm_cuda` execution plan and fallback rejection | Model-specific artifact, tokenizer, prompt policy, answer-quality receipt |
| Product CLI readiness | Model route receipt plus normal `model status`, `model verify`, `ask`, `chat`/warm-session, `bench` review, and `receipts explain` surfaces | Model coverage row alignment |
| Exact-profile server readiness | `server_shared_engine_cuda` plus underlying model route and server receipt fields | Endpoint/profile/readiness scope |
| Speedup | CUDA route receipt plus exact same-artifact CPU comparator | Accepted benchmark qualification decision |
| Full residency | CUDA route receipt plus phase-by-phase residency proof | Runtime performance residency contract |

No CUDA route receipt may promote speedup, full residency, broad server
readiness, or another model-family proof unless the applicable spec and receipt
fields separately accept that exact claim.

## Receipt Explanation Requirements

`bitnet receipts explain` and any model status surface that summarizes CUDA
receipts must expose:

- model coverage row when known;
- current support tier when known;
- requested backend;
- selected backend;
- runtime API;
- selected route;
- proof-family booleans;
- fallback status;
- execution-plan counts;
- speedup status;
- server readiness status and scope;
- residency status;
- forbidden claims.

Missing model matrix data should degrade gracefully, but it must not infer a
stronger route, proof family, support tier, speedup, server, or residency claim.

## Non-Goals

- Do not implement runtime routing, kernels, CLI behavior, server behavior, or
  receipt validators in this spec.
- Do not promote any model coverage row.
- Do not edit hardware receipts, generated dashboards, CI workflows, policy
  ledgers, or model manifests.
- Do not require CUDA hardware proof in ordinary PR CI.
- Do not claim speedup, full residency, broad server readiness, broad chat
  quality, or cross-model inheritance.

## Proof Commands

Docs-only changes to this spec should run:

```bash
cargo run --locked -p xtask --no-default-features -- check-model-coverage
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

If a future runtime PR implements this contract, it must add the exact command
that produced the CUDA receipt and the command that explains or validates it.
