# BITNET-SPEC-I2S-CUDA

Status: active
Owner: BitNet-rs maintainers
Created: 2026-05-19
Linked proposal: docs/proposals/BITNET-PROP-0015-i2s-productization.md
Linked specs:
  docs/specs/BITNET-SPEC-CUDA-ROUTE-CONTRACT.md,
  docs/specs/BITNET-SPEC-I2S-KERNEL-IDENTITY.md,
  docs/specs/BITNET-SPEC-I2S-STATUS-SURFACE.md,
  docs/specs/BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md,
  docs/specs/BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE.md,
  docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md
Linked ADRs: docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md
Linked plan: plans/i2s/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: yes
Policy impact: no

This spec defines the route-specific contract for production BitNet I2_S/QK256
CUDA claims. It narrows the common CUDA route contract to the official
Microsoft I2_S GGUF row and prevents diagnostic QK256, dense CUDA, generic
CUDA, or server-smoke evidence from being promoted into broader BitNet claims.

## Purpose

The I2_S CUDA lane exists to prove that the official packed BitNet artifact can
execute BitNet QK256 linear work on the selected CUDA backend with hidden
fallback rejected. It is not a dense SLM route, a generic GPU route, or a
marketing speedup claim.

Production I2_S CUDA proof requires all of these to be explicit in the receipt
or in a linked model coverage row:

- exact model artifact identity;
- tokenizer and prompt authority;
- strict selected backend and runtime API;
- selected route;
- selected production QK256 CUDA kernel identity or kernel-stat alias;
- route-relevant CUDA invocation counts;
- CPU fallback and unsupported-op counts;
- proof-family booleans;
- claim booleans for speed, residency, and server readiness.

## Canonical Artifact And Route

The production BitNet I2_S CUDA row is:

```text
model_coverage_row = bitnet_official_2b_i2s_qk256
model repo         = microsoft/bitnet-b1.58-2B-4T-gguf
model file         = ggml-model-i2_s.gguf
artifact kind      = gguf_i2_s
tokenizer          = external_llama_bpe
prompt authority   = bitnetcpp-answer
selected backend   = nvidia-rtx-5070-ti-cuda
runtime API        = cuda
selected route     = bitnet_qk256_cuda
proof family       = BitNet packed I2_S/QK256 CUDA
```

`requested_backend="cuda"` is only a user selector. A strict RTX 5070 Ti proof
must resolve it to `selected_backend="nvidia-rtx-5070-ti-cuda"` before it may
claim the I2_S CUDA lane.

## Kernel Identity

Production I2_S CUDA claims require an explicit production QK256 CUDA kernel
identity. Receipts may use either the canonical kernel identity from
`BITNET-SPEC-I2S-KERNEL-IDENTITY` or the current hardware receipt alias:

```text
canonical selected kernel = qk256-cuda-i8s-scaled-gemv
hardware receipt alias    = qk256_gemv_cuda
```

The receipt must include a route-relevant invocation count greater than zero.
Examples:

```text
qk256_gemv_cuda invocations > 0
bitnet_qk256_cuda_ops > 0
linear_layers_on_cuda > 0
```

Diagnostic F32/no-scale QK256 kernels, tiny CUDA smoke kernels, visibility
probes, and generic CUDA kernels do not satisfy production packed I2_S/QK256
proof.

## Required Receipt Fields

Any receipt used to support `bitnet_packed_i2s_qk256_proof=true`,
`accelerator_answer_ready=true`, `product_cli_ready=true`, benchmark review, or
server smoke for the I2_S lane must expose these fields or equivalent nested
fields that `receipts explain` can normalize:

```json
{
  "model_coverage_row": "bitnet_official_2b_i2s_qk256",
  "requested_backend": "cuda | nvidia-rtx-5070-ti-cuda",
  "selected_backend": "nvidia-rtx-5070-ti-cuda",
  "runtime_api": "cuda",
  "selected_route": "bitnet_qk256_cuda",
  "fallback_used": false,
  "fallback_reason": null,
  "execution_plan": {
    "route": "bitnet_qk256_cuda",
    "bitnet_qk256_cuda_ops": 1,
    "dense_regular_llm_cuda_ops": 0,
    "cpu_fallback_ops": 0,
    "unsupported_ops": 0
  },
  "proof_family": {
    "bitnet_packed_i2s_qk256_proof": true,
    "dense_regular_llm_cuda_proof": false
  },
  "claims": {
    "speedup_claim": false,
    "full_residency_claim": false,
    "server_ready": false
  }
}
```

Older receipts may preserve these facts under `bitnet`, `execution_coverage`,
`kernel_stats`, `model`, or `claim_boundary`. Normalizers must not infer a true
claim from missing fields.

## Proof Ladder

The I2_S CUDA row follows the common model promotion ladder:

| Tier | I2_S CUDA entry requirement |
| --- | --- |
| `registered` | Official artifact row exists with model family, artifact kind, verifier surface, and forbidden claims. |
| `structurally_valid` | Artifact contract and tokenizer/prompt authority are recorded. |
| `reference_good` | CPU/reference answer sanity passes with the same artifact and prompt policy. |
| `cpu_answer_ready` | Normal CPU user path can answer with fallback and quality state recorded. |
| `accelerator_answer_ready` | Strict CUDA route executes with `fallback_used=false`, route invocation evidence, and answer quality evidence. |
| `benchmark_qualified` | Exact benchmark profile is reviewed under the runtime performance contract. |
| `product_cli_ready` | Normal `ask`/`chat`/receipt paths are accepted for the exact model/backend/route while unsupported claims remain false. |
| `server_ready` | Server readiness is separately reviewed for an exact endpoint/profile or a broader scope. CLI readiness does not imply this tier. |

Current official BitNet I2_S/QK256 status is `product_cli_ready` with
`benchmark_qualified=false`, `server_ready=false`, `speedup_claim=false`, and
`full_residency_claim=false`.

## Accepted Proof Profiles

Receipts may support the I2_S CUDA lane only when they use the same official
artifact, tokenizer authority, prompt policy, selected backend, and selected
route. Accepted profile families are:

```text
artifact verification
prompt/tokenizer authority audit
CPU reference or CPU answer corpus
one_token_strict_cuda
short_decode_strict_cuda
warm_session
strict_cuda_ask
benchmark_profile_receipt
server_shared_engine_chat_completion smoke
```

A benchmark receipt remains a benchmark baseline until exact-profile speed
review accepts it. A server smoke receipt remains smoke until server readiness
review accepts an exact-profile or broad readiness scope.

## Claim Booleans

For the official I2_S CUDA row:

```text
bitnet_packed_i2s_qk256_proof = true
dense_regular_llm_cuda_proof = false
speedup_claim = false unless exact-profile review accepts speedup
full_residency_claim = false unless every required residency phase proves it
server_ready = false unless server readiness review accepts the scoped profile
```

`bitnet_packed_i2s_qk256_proof=true` requires the exact official artifact,
`selected_route=bitnet_qk256_cuda`, strict selected backend identity, production
QK256 CUDA kernel evidence, and rejected fallback. It is not inherited by
Falcon, TL1/TL2, GPU-int2, dense Qwen, Qwen3, SmolLM2, Llama, Gemma, Phi,
OpenCL, OpenVINO, ROCm, Metal, WGPU, Vulkan, or CPU lanes.

## Hard Rails

- Dense regular-LLM CUDA proof must not satisfy BitNet packed I2_S/QK256 proof.
- BitNet packed I2_S/QK256 proof must not satisfy dense regular-LLM CUDA proof.
- Generic `cuda` without strict selected-backend resolution is not RTX 5070 Ti proof.
- CPU fallback, missing execution-plan counts, or unsupported strict CUDA ops reject accelerator claims.
- Tiny kernel smoke, device visibility, or NVML visibility is not model-route execution proof.
- Diagnostic F32/no-scale QK256 parity is not production packed I2_S/QK256 proof.
- Server smoke is not broad server readiness.
- Benchmark timing is not speedup unless exact-profile review accepts it.
- Upload-once weights or QK256 linears alone do not prove full model residency.

## Rejection Examples

| Evidence | Result |
| --- | --- |
| `selected_backend="cuda"` with no strict RTX 5070 Ti label | Reject strict backend proof. |
| `selected_route="dense_regular_llm_cuda"` | Reject BitNet I2_S/QK256 proof. |
| `fallback_used=true` or `cpu_fallback_ops > 0` | Reject accelerator proof. |
| `qk256_gemv_cuda` absent or zero invocations for an execution claim | Reject production QK256 CUDA proof. |
| Tiny CUDA smoke kernel only | Keep as probe/smoke, not I2_S route proof. |
| Server response receipt with no BitNet route execution evidence | Keep as server smoke or diagnostic, not route proof. |
| Benchmark receipt with `speedup_claim=true` and no exact CPU comparator profile | Reject speedup claim. |

## Acceptance

This spec is accepted when:

- it names the official artifact, route, backend, and kernel identities;
- it defines required receipt fields for strict I2_S CUDA proof;
- it defines criteria for `bitnet_packed_i2s_qk256_proof=true`;
- it keeps dense CUDA proof, speedup, full residency, and server readiness
  separate;
- it repairs the incomplete I2_S hard-rule text without changing runtime
  behavior or model coverage rows.

## Proof Commands

Docs-only validation:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- check-model-coverage
```

If campaign files, generated dashboards, model coverage rows, or hardware
receipts are edited in the same PR, also run their listed generation/check
commands. This spec change alone does not require model downloads or hardware
runs.

## Non-Goals

- Do not change runtime math, kernels, tokenizer, loader, server behavior, or
  benchmark logic.
- Do not promote speedup, full residency, server readiness, or broad CUDA
  support.
- Do not add new model families to the official I2_S proof row.
- Do not make diagnostic QK256 parity or dense CUDA evidence product BitNet
  proof.
