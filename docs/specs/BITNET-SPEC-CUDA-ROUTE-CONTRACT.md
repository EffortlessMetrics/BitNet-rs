# BITNET-SPEC-CUDA-ROUTE-CONTRACT: CUDA Route Contract

Status: Draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0002](../proposals/BITNET-PROP-0002-9950x3d-5070ti-cuda-productization.md),
  [BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
Linked specs: [BITNET-SPEC-0007](BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md),
  [BITNET-SPEC-0013](BITNET-SPEC-0013-model-onboarding-proof-ladder.md),
  [BITNET-SPEC-0014](BITNET-SPEC-0014-runtime-performance-contract.md),
  [BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE](BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE.md)
Linked ADRs: [BITNET-ADR-0004](../adr/BITNET-ADR-0004-9950x3d-5070ti-cuda-product-bench.md)
Linked plan: [CUDA 5070 Ti productization](../../plans/cuda-5070ti-productization/README.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines route and proof-family fields required before
  CUDA receipts can promote accelerator, product, server, speed, or residency
  claims.
Policy impact: No CI policy change; CUDA hardware proof remains outside
  ordinary PR CI.

## Purpose

This spec defines the route vocabulary and receipt fields that make CUDA claims
machine-checkable for BitNet-rs. It narrows the existing CUDA product contract
under BITNET-SPEC-0007 and the model proof ladder under BITNET-SPEC-0013 so
users and maintainers can tell which CUDA route actually executed and which
proof family, if any, the receipt can support.

A CUDA receipt must never let these proof families look interchangeable:

- official BitNet I2_S/QK256 CUDA;
- dense regular-LLM CUDA for dense SLMs or small dense LLMs;
- dense GGUF diagnostic parity fixtures;
- dense GGUF layer-planning evidence;
- server shared-engine CUDA execution;
- CPU reference execution.

## Scope

This spec applies to CUDA receipts, CUDA model coverage rows, and user-facing
surfaces that summarize CUDA receipts, including:

```text
bitnet model status --device nvidia-rtx-5070-ti-cuda
bitnet model verify <model>
bitnet ask --device cuda --model <model> "..."
bitnet chat --device cuda --model <model>
bitnet bench --device cuda --model <model>
bitnet serve --device cuda --model <model>
bitnet receipts explain --latest --format json
```

It is a documentation and receipt contract. It does not implement runtime
routing, kernels, CLI behavior, server behavior, or benchmark promotion.

## Route IDs

CUDA receipts and status explanations must use one of these route IDs when the
receipt is claiming or evaluating CUDA execution:

| Route ID | Meaning | May prove | Must not prove |
| --- | --- | --- | --- |
| `bitnet_qk256_cuda` | Official BitNet I2_S/QK256 route that executes packed QK256 BitNet linear work through CUDA kernels. | BitNet packed I2_S/QK256 CUDA for the exact artifact, backend, and profile. | Dense regular-LLM CUDA, TL1/TL2, GPU-int2 master routes, speedup, full residency, or broad server readiness. |
| `dense_regular_llm_cuda` | Dense SLM or small dense LLM route that executes dense regular-LLM model work through CUDA. | Dense regular-LLM CUDA for the exact model family, artifact, backend, and profile. | BitNet I2_S/QK256, 1-bit packed-kernel proof, or another dense model family's proof. |
| `dense_gguf_linear_cuda_parity` | Diagnostic dense GGUF single-linear or role-sweep parity fixture. | The scoped parity fixture only. | Whole-model CUDA execution, answer readiness, product CLI readiness, server readiness, speedup, or full residency. |
| `dense_gguf_layer_plan` | Dense GGUF all-layer route-planning descriptor with supported and unsupported op counts. | Planning readiness and missing-op accounting. | CUDA execution, answer quality, product CLI readiness, server readiness, speedup, or residency. |
| `server_shared_engine_cuda` | Server path using a shared CUDA engine for a named endpoint/profile. | Exact-profile server readiness only when paired with endpoint, profile, model row, fallback, and response-quality proof. | CLI readiness, streaming/concurrency/long-context readiness outside the proven profile, speedup, full residency, or other model-family proof. |

Receipts may include narrower sub-route or kernel fields, but those fields must
not replace the route ID above when promoting a CUDA support claim.

## Required CUDA Receipt Fields

Every CUDA receipt that can be used for accelerator, product, server,
benchmark, speed, or residency claims must include enough structured data to
explain backend selection, fallback, route identity, execution-plan counts, and
proof-family booleans.

Minimum shape:

```json
{
  "requested_backend": "cuda | nvidia-rtx-5070-ti-cuda",
  "selected_backend": "nvidia-rtx-5070-ti-cuda",
  "runtime_api": "cuda",
  "selected_route": "bitnet_qk256_cuda | dense_regular_llm_cuda | dense_gguf_linear_cuda_parity | dense_gguf_layer_plan | server_shared_engine_cuda",
  "fallback_used": false,
  "fallback_reason": null,
  "execution_plan": {
    "route": "bitnet_qk256_cuda",
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

A route-specific receipt may add fields such as kernel invocation counts,
server request IDs, timing data, transfer counters, or residency phase tables.
Those fields refine the claim but do not remove the minimum fields above.

## Backend Selection Rules

- `requested_backend="cuda"` is a convenience selector only. A CUDA proof
  receipt must resolve it to `selected_backend="nvidia-rtx-5070-ti-cuda"`
  before claiming RTX 5070 Ti CUDA execution.
- `selected_backend="cuda"` is not a strict product proof value.
- `runtime_api="cuda"` is required for CUDA proof. WGPU, Vulkan, D3D12,
  OpenCL, ROCm, Metal, and CPU receipts must keep their own API identities.
- Hardware visibility, device enumeration, NVML presence, and a tiny kernel
  smoke are not whole-route execution proof unless paired with a route and
  execution plan.

## Execution Plan Rules

A CUDA execution plan must make hidden fallback and unsupported work visible.
At minimum it must report:

- the route being evaluated;
- BitNet QK256 CUDA op count;
- dense regular-LLM CUDA op count;
- CPU fallback op count;
- unsupported op count.

For strict CUDA product claims:

- `fallback_used` must be `false`;
- `fallback_reason` must be `null`;
- `cpu_fallback_ops` must be `0`;
- the route-relevant CUDA op count must be greater than `0` for execution
  claims;
- a receipt with no execution plan cannot promote a CUDA claim.

## Proof Family Rules

Proof family booleans must stay explicit and non-interchangeable:

| Claim boolean | May be true when | Must be false when |
| --- | --- | --- |
| `bitnet_packed_i2s_qk256_proof` | The exact official BitNet I2_S/QK256 artifact executed the `bitnet_qk256_cuda` route with route-specific proof and fallback rejected. | The route is dense regular-LLM, dense parity-only, planning-only, server-only without BitNet route proof, CPU, WGPU, Vulkan, or generic hardware smoke. |
| `dense_regular_llm_cuda_proof` | The exact dense model artifact executed the `dense_regular_llm_cuda` route with model-family proof and fallback rejected. | The route is BitNet QK256, dense parity-only, planning-only, CPU, WGPU, Vulkan, or another dense model's proof. |

A receipt may set both booleans to `false` for diagnostic, planning,
benchmark-review, or server receipts that do not independently prove either
family. A receipt must not set both to `true` unless a later spec defines an
explicit composite route and its independent evidence requirements.

## Claim Rails

CUDA route receipts must enforce these rails:

- Dense CUDA can never satisfy BitNet packed I2_S/QK256 proof.
- BitNet QK256 CUDA can never satisfy dense regular-LLM CUDA proof.
- CPU AVX-512 receipts are same-box reference evidence, not CUDA execution.
- Generic `cuda` without strict backend resolution is not RTX 5070 Ti proof.
- Strict CUDA with CPU fallback is a hard failure for CUDA execution claims.
- Dense GGUF parity fixtures are diagnostics, not product route proof.
- Dense GGUF layer plans are route-planning evidence, not execution proof.
- Exact-profile server readiness does not imply CLI readiness, streaming
  readiness, concurrency readiness, speedup, or full residency.
- Benchmark qualification is exact-profile only and requires the runtime
  performance contract fields before any speed claim.
- Upload-once weights, kernel invocation counts, or CUDA linears alone do not
  prove full model residency.

## Status And Receipt Explanation Requirements

`model status` and `receipts explain` summaries for CUDA claims should expose:

- model coverage row;
- current support tier;
- requested backend;
- selected backend;
- runtime API;
- selected route;
- proof family booleans;
- fallback status;
- execution-plan counts;
- server readiness and scope when applicable;
- speedup claim status;
- full-residency claim status;
- forbidden claims that remain false.

Missing model coverage rows or missing optional detail should degrade to an
honest unknown or unsupported explanation. They must not silently promote the
receipt to CUDA product support.

## Acceptance

This spec is accepted when:

- the route IDs above are documented and linked from the model proof ladder,
  CUDA capability matrix, and CUDA productization plan;
- receipt fields and proof-family booleans are defined;
- hard rails prevent dense/BitNet proof-family conflation;
- no runtime behavior changes are made;
- no model coverage row is promoted.

## Proof Commands

Docs-only validation for this spec:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- check-model-coverage
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
```

If a status page with its own validation block is edited in the same PR, also
run its listed checker or record why it is unavailable.

## Non-Goals

- Do not implement CUDA kernels or runtime dispatch in this spec.
- Do not change model coverage tiers or claim booleans in this spec.
- Do not add or edit hardware proof receipts in this spec.
- Do not define BitNet QK256 kernel semantics; a narrower BitNet CUDA spec owns
  that contract.
- Do not define dense model-family onboarding details; a narrower dense CUDA
  spec owns that contract.
- Do not define full residency phase proof; a narrower residency spec owns that
  contract.
- Do not define exact-profile server readiness; a narrower server spec owns
  that contract.
