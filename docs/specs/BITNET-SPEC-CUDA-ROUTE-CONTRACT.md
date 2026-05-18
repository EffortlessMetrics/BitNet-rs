# BITNET-SPEC-CUDA-ROUTE-CONTRACT: CUDA Route Contract

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal:
[BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md),
[BITNET-PROP-0002](../proposals/BITNET-PROP-0002-9950x3d-5070ti-cuda-productization.md)
Linked specs:
[BITNET-SPEC-0013](BITNET-SPEC-0013-model-onboarding-proof-ladder.md),
[BITNET-SPEC-0014](BITNET-SPEC-0014-runtime-performance-contract.md),
[BITNET-SPEC-0007](BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md)
Linked ADRs:
[BITNET-ADR-0004](../adr/BITNET-ADR-0004-9950x3d-5070ti-cuda-product-bench.md),
[BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan:
[9950X3D + RTX 5070 Ti CUDA Productization Plan](../../plans/cuda-5070ti-productization/README.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines CUDA route receipt fields and proof-family rails;
does not promote any model coverage row.
Policy impact: n/a

## Purpose

CUDA support in BitNet-rs is not one interchangeable proof family. A receipt
must identify the selected backend, runtime API, route, execution plan,
fallback status, and proof family before it can support a CUDA claim.

This spec defines the narrow route vocabulary and receipt contract used by CUDA
model-status, ask/chat, bench, server, and receipt-explanation surfaces. It is a
child contract under the model onboarding proof ladder and runtime performance
contract; it does not add new runtime behavior or promote any model.

## Route IDs

CUDA receipts that make route claims must use one of these route IDs.

| Route ID | Meaning | Claim boundary |
| --- | --- | --- |
| `bitnet_qk256_cuda` | Official BitNet I2_S/QK256 packed route using CUDA QK256 model math evidence. | Proves only the scoped BitNet packed I2_S/QK256 route for the named artifact/backend/profile. |
| `dense_regular_llm_cuda` | Dense SLM or small dense LLM CUDA inference route for a named model family and artifact. | Proves only that dense model row and profile; it is not BitNet proof. |
| `dense_gguf_linear_cuda_parity` | Dense GGUF single-linear or boundary parity fixture. | Fixture evidence only; it is not full model CUDA readiness. |
| `dense_gguf_layer_plan` | Dense GGUF all-layer execution-plan or unsupported-op accounting. | Planning evidence only; it is not CUDA execution proof. |
| `server_shared_engine_cuda` | Shared-engine CUDA server route for a named endpoint/profile. | Server-profile evidence only; it does not imply broad serving, speedup, or full residency. |

Future CUDA routes must be added to this table before receipts use them for
promotion.

## Required CUDA Receipt Fields

Every CUDA receipt that supports a user-facing claim must include these fields
or an explicitly documented equivalent path in the receipt schema:

```json
{
  "requested_backend": "cuda | nvidia-rtx-5070-ti-cuda",
  "selected_backend": "nvidia-rtx-5070-ti-cuda",
  "runtime_api": "cuda",
  "selected_route": "bitnet_qk256_cuda | dense_regular_llm_cuda",
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

Receipts may include additional route-specific fields, but these common fields
are the minimum needed to prevent hidden fallback and proof-family conflation.

## Backend Resolution

`cuda` is a user convenience selector, not proof by itself. A CUDA receipt may
record `requested_backend = "cuda"` only if it resolves the actual selected
backend before making any proof claim.

For the 9950X3D + RTX 5070 Ti lane, strict proof receipts must resolve to:

```text
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
fallback_used = false
```

A receipt that leaves `selected_backend` as generic `cuda` cannot promote RTX
5070 Ti CUDA status.

## Proof-Family Rules

CUDA proof families are non-interchangeable:

- Dense regular-LLM CUDA proof never satisfies BitNet packed I2_S/QK256 proof.
- BitNet packed I2_S/QK256 proof never satisfies dense regular-LLM CUDA proof.
- Dense linear parity fixtures do not prove dense full-model CUDA readiness.
- Dense all-layer plans do not prove CUDA execution.
- Server shared-engine CUDA receipts prove only the named server endpoint,
  streaming mode, request profile, model row, backend, route, and readiness
  scope.
- CPU AVX-512, WGPU, OpenCL, Vulkan, Metal, NPU, hardware visibility, and CUDA
  probe receipts do not prove CUDA model execution.

## Fallback Rules

Strict CUDA receipts must reject hidden fallback:

- `fallback_used = true` is a hard failure for strict CUDA proof.
- `cpu_fallback_ops > 0` is a hard failure for strict CUDA proof unless the
  receipt is explicitly a diagnostic non-promotion receipt.
- `unsupported_ops > 0` may be valid planning evidence, but it blocks route
  promotion until the unsupported operations are resolved or scoped out by an
  accepted spec.
- A missing fallback field cannot be interpreted as fallback-free.

## Execution-Plan Rules

A CUDA receipt with no execution plan cannot promote a CUDA claim. The execution
plan must state the route and route-family operation counters so reviewers can
answer which CUDA route actually executed.

For BitNet QK256 receipts, `bitnet_qk256_cuda_ops` or route-specific QK256
kernel counters must be greater than zero for execution proof.

For dense regular-LLM CUDA receipts, `dense_regular_llm_cuda_ops` or equivalent
dense route counters must be greater than zero for execution proof.

## Claim Booleans

CUDA model coverage rows and receipt explanations must keep proof booleans
separate:

```json
{
  "bitnet_packed_i2s_qk256_proof": false,
  "dense_regular_llm_cuda_proof": true,
  "server_ready": false,
  "speedup_claim": false,
  "full_residency_claim": false
}
```

A row may set only the proof booleans supported by matching receipts. Speedup,
server readiness, and full residency remain separate claims even when CUDA
execution and answer quality are proven.

## Required Explanation Surface

`bitnet receipts explain` and `bitnet model status` should expose enough route
contract information for users to distinguish:

- requested backend versus selected backend;
- generic CUDA selector versus strict RTX 5070 Ti proof;
- BitNet QK256 CUDA versus dense regular-LLM CUDA;
- execution proof versus planning or fixture evidence;
- fallback-free execution versus rejected fallback;
- product CLI readiness versus server readiness;
- benchmark qualification versus unqualified speed;
- upload-once or linear residency versus full model residency.

## Acceptance

This spec is accepted when:

- CUDA route IDs are explicit and reusable by later receipt validators;
- required common receipt fields are defined;
- proof-family boundaries are stated as hard rails;
- strict CUDA fallback rules are documented;
- no runtime behavior changes are introduced by this docs PR;
- no model coverage row is promoted by this docs PR.

## Non-Goals

This spec does not prove or implement:

- a new CUDA kernel;
- a new BitNet QK256 answer receipt;
- a new dense SLM answer receipt;
- speedup;
- full CUDA residency;
- server readiness;
- broad chat quality;
- support for a GPU other than the selected proof backend named in a receipt.

## Validation

For docs-only edits to this route contract and linked status surfaces, run:

```bash
git diff --check
```

When this contract is wired into generated model coverage or receipt validators,
those later PRs must also run the relevant generator or checker named by the
plan item.
