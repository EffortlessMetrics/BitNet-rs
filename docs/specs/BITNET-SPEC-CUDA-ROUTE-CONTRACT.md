# BITNET-SPEC-CUDA-ROUTE-CONTRACT: CUDA Route Contract

Status: proposed
Owner: cuda/product
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0002](../proposals/BITNET-PROP-0002-9950x3d-5070ti-cuda-productization.md), [BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
Linked specs: [BITNET-SPEC-0007](BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md), [BITNET-SPEC-0010](BITNET-SPEC-0010-server-readiness-proof-boundary.md), [BITNET-SPEC-0013](BITNET-SPEC-0013-model-onboarding-proof-ladder.md), [BITNET-SPEC-0014](BITNET-SPEC-0014-runtime-performance-contract.md)
Linked ADRs: [BITNET-ADR-0004](../adr/BITNET-ADR-0004-9950x3d-5070ti-cuda-product-bench.md)
Linked plan: [9950X3D + RTX 5070 Ti CUDA Productization Plan](../../plans/cuda-5070ti-productization/README.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines route and proof-family fields only; no tier or model-coverage promotion.
Policy impact: No CI or policy exception change.

## Purpose

CUDA support must be explainable by route, model family, backend identity, and
receipt evidence. A CUDA receipt that proves one route must not be reused to
promote a different route, a different model family, speedup, server readiness,
or full residency.

This spec defines the minimum route vocabulary and receipt fields required
before CUDA user-facing surfaces may say which CUDA path executed. It narrows
and composes the existing 9950X3D + RTX 5070 Ti product contract,
server-readiness boundary, model-onboarding ladder, and runtime-performance
contract; it does not replace them.

## Scope

This contract applies to CUDA receipts, model coverage rows, model status,
`ask`, `chat`, `bench`, `serve`, and `receipts explain` surfaces when they make
or explain CUDA execution claims.

It covers route identity and proof-family separation for:

- official Microsoft BitNet 2B I2_S/QK256 CUDA;
- dense SLM CUDA such as Qwen2.5 and Qwen3;
- dense GGUF CUDA fixture and layer-plan proof;
- shared-engine CUDA server profiles.

## Required Route IDs

CUDA proof receipts and explanations must use the narrowest applicable route
identifier from this governed vocabulary:

| Route ID | Meaning | May prove | Must not prove |
| --- | --- | --- | --- |
| `bitnet_qk256_cuda` | Official BitNet packed I2_S/QK256 route using CUDA QK256 execution evidence. | BitNet packed I2_S/QK256 CUDA for the exact artifact and profile. | Dense regular-LLM CUDA, TL1/TL2, GPU-int2 master-route, speedup, full residency, or broad server readiness. |
| `dense_regular_llm_cuda` | Dense regular-LLM CUDA route for an exact dense SLM or small dense LLM artifact. | Dense CUDA execution for the exact model row and profile. | BitNet I2_S/QK256 proof or another dense model row's proof. |
| `dense_gguf_linear_cuda_parity` | Dense GGUF single-linear or fixture parity route. | Narrow linear parity for the tested descriptor and fixture. | Whole-model dense CUDA readiness, BitNet proof, speedup, server readiness, or full residency. |
| `dense_gguf_layer_plan` | Dense GGUF all-layer planning route with unsupported-op accounting. | Planning completeness or blockers for the exact artifact. | CUDA execution, answer readiness, speedup, server readiness, or full residency. |
| `server_shared_engine_cuda` | Server request used the shared CUDA inference engine for an exact endpoint/profile. | Server readiness only for the exact model, endpoint, streaming mode, and profile when the route also names the model-family proof. | Broad serving readiness, streaming/concurrency readiness without matching proof, speedup, full residency, or cross-family proof. |

New CUDA route IDs require a later spec or an update to this spec before a
model coverage row or receipt explanation may promote them.

## Required CUDA Receipt Fields

Every CUDA receipt that supports an execution claim must include these fields or
an equivalent machine-readable representation with the same meaning:

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

The selected route and proof-family booleans must agree. For example,
`bitnet_qk256_cuda` receipts that claim BitNet proof must keep
`dense_regular_llm_cuda_proof=false`, and dense CUDA receipts must keep
`bitnet_packed_i2s_qk256_proof=false`.

## Backend Resolution

`cuda` may be accepted as a user convenience selector. It is not a strict proof
identity until the receipt resolves it to the selected backend.

For the current CUDA product bench, strict execution claims require:

```text
requested_backend = cuda | nvidia-rtx-5070-ti-cuda
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
fallback_used = false
```

A receipt whose selected backend remains generic `cuda` cannot promote RTX 5070
Ti CUDA proof.

## Proof-Family Rails

CUDA proof families are non-interchangeable:

- dense CUDA can never satisfy BitNet packed I2_S/QK256 proof;
- BitNet QK256 CUDA can never satisfy dense regular-LLM CUDA proof;
- CPU AVX-512 can provide reference evidence but not CUDA execution proof;
- WGPU, Vulkan, D3D12, OpenCL, Metal, NPU, generic GPU, or hardware-visibility
  receipts cannot satisfy CUDA proof;
- a CUDA receipt with no execution plan cannot promote a CUDA execution claim;
- fallback to CPU under strict CUDA is a hard failure, not a successful CUDA
  result.

## Claim Boundaries

A valid route receipt may support only the claim family it proves:

| Receipt evidence | May claim | Must not claim |
| --- | --- | --- |
| `bitnet_qk256_cuda` with QK256 kernel evidence and fallback rejection | Official BitNet I2_S/QK256 CUDA execution for the exact artifact/profile. | Dense CUDA, global speedup, full residency, broad server readiness, broad chat quality. |
| `dense_regular_llm_cuda` with exact artifact evidence and fallback rejection | Dense regular-LLM CUDA execution for that exact model/profile. | BitNet packed I2_S/QK256, another dense model, global speedup, full residency. |
| `dense_gguf_linear_cuda_parity` | Linear fixture parity only. | Whole-model answer readiness or product CLI readiness. |
| `dense_gguf_layer_plan` | Layer-plan completeness and unsupported-op counts. | CUDA execution or answer quality. |
| `server_shared_engine_cuda` with endpoint/profile fields | Exact-profile server route evidence for the same model-family proof. | Streaming, concurrency, long-context, broad production, speedup, or full residency without separate proof. |

Speedup, server readiness, and full residency remain separate claim families.
They require the governing benchmark, server, or residency fields before status
surfaces or model coverage rows may promote them.

## User-Facing Explanation Requirements

`bitnet receipts explain`, `bitnet model status`, status docs, and model
coverage summaries must expose enough route information to answer:

- what backend was requested and what backend was selected;
- which CUDA route executed;
- which proof family is true and which proof families are false;
- whether fallback was rejected;
- which execution-plan counters support the route;
- which claims remain forbidden for this receipt.

If a receipt lacks the model coverage matrix row or a route cannot be matched,
the explanation must degrade to a narrower diagnostic claim instead of
promoting CUDA readiness.

## Validation

This is a documentation-only spec. Validation for changes to this spec is:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- check-model-coverage
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
```

Runtime PRs that implement validators or receipt explanations must add their
own crate-level tests and fixture checks for the route and proof-family rails in
this spec.

## Non-Goals

- Do not implement runtime CUDA routing in this spec.
- Do not modify existing receipts in this spec.
- Do not promote any model coverage row in this spec.
- Do not claim CUDA speedup, full residency, server readiness, or broad chat
  quality from route identity alone.
- Do not make CUDA hardware proof part of ordinary PR CI.
