# BITNET-SPEC-I2S-CUDA: I2_S/QK256 CUDA Contract

Status: active
Owner: BitNet-rs maintainers
Created: 2026-05-19
Linked proposal: [BITNET-PROP-0015](../proposals/BITNET-PROP-0015-i2s-productization.md)
Linked specs:
[BITNET-SPEC-0007](BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md),
[BITNET-SPEC-0013](BITNET-SPEC-0013-model-onboarding-proof-ladder.md),
[BITNET-SPEC-0014](BITNET-SPEC-0014-runtime-performance-contract.md),
[BITNET-SPEC-CUDA-ROUTE-CONTRACT](BITNET-SPEC-CUDA-ROUTE-CONTRACT.md),
[BITNET-SPEC-I2S-QK256-LAYOUT](BITNET-SPEC-I2S-QK256-LAYOUT.md),
[BITNET-SPEC-I2S-KERNEL-IDENTITY](BITNET-SPEC-I2S-KERNEL-IDENTITY.md),
[BITNET-SPEC-I2S-STATUS-SURFACE](BITNET-SPEC-I2S-STATUS-SURFACE.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [I2_S implementation plan](../../plans/i2s/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines the route-specific proof contract for BitNet
  packed I2_S/QK256 CUDA claims.
Policy impact: no

## Purpose

This spec defines the CUDA proof contract for production I2_S/QK256 BitNet
execution. It narrows the common CUDA route contract to the BitNet packed
QK256 route so receipts, model status, and support bundles can distinguish
official BitNet CUDA proof from dense regular-LLM CUDA, diagnostic QK256
fixtures, hardware visibility, and CPU fallback.

## Scope

This spec applies to receipts and status rows that claim BitNet packed
I2_S/QK256 CUDA execution for an exact model artifact, backend, and profile.
The first product lane is the official Microsoft BitNet 2B I2_S/QK256 artifact
on the 9950X3D + RTX 5070 Ti box.

This spec does not promote new support. It defines what future runtime,
benchmark, server, and status PRs must prove before setting BitNet CUDA claim
booleans.

## Hard Rule

Production I2_S/QK256 CUDA claims require an exact model artifact, tokenizer
and prompt authority, strict selected backend, CUDA runtime API, selected
BitNet route, production QK256 kernel identity, kernel invocation counters,
fallback rejection, zero unsupported strict ops, zero BitNet linear CPU
fallback, and receipt/status claim booleans that keep dense SLM proof false.

## Route Identity

Production BitNet CUDA receipts must use:

```text
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
selected_route = bitnet_qk256_cuda
```

`requested_backend="cuda"` is only a selector. It is not a strict proof value
until the receipt resolves it to the selected backend above.

## Production And Diagnostic Boundaries

| Evidence | May prove | Must not prove |
| --- | --- | --- |
| Production packed I2_S/QK256 CUDA route with fallback rejected | `bitnet_packed_i2s_qk256_proof=true` for the exact artifact/profile | Dense SLM CUDA, speedup, full residency, broad server readiness |
| F32 or no-scale QK256 diagnostic kernels | Diagnostic parity or kernel debugging only | Production packed I2_S/QK256 proof |
| CUDA device visibility or kernel smoke | Hardware/runtime availability | Model route execution, answer quality, speedup |
| CPU AVX-512 comparator | Same-box reference or quality comparator | CUDA execution proof |
| Dense regular-LLM CUDA route | Dense model-family proof for that exact dense artifact | BitNet packed I2_S/QK256 proof |

Diagnostic receipts must keep `bitnet_packed_i2s_qk256_proof=false` unless
they also include independent production packed I2_S/QK256 route evidence.

## Required Receipt Fields

Any receipt used to support a BitNet CUDA claim must include these fields or an
explicit not-applicable reason when a field is profile-specific:

```text
model_artifact
model_coverage_row
tokenizer_authority
prompt_template
requested_backend
selected_backend
runtime_api
selected_route
fallback_used
fallback_reason
unsupported_strict_ops
bitnet_linear_cpu_fallback_ops
qk256_cuda_ops
selected_kernel_id
qk256_gemv_cuda_invocations
weight_upload_once
quality_gate
bitnet_packed_i2s_qk256_proof
dense_regular_llm_cuda_proof
speedup_claim
full_residency_claim
server_ready
receipt_id
receipt_path
```

Benchmark, TTFT, throughput, residency, and server receipts must also satisfy
the runtime performance contract fields in
[BITNET-SPEC-0014](BITNET-SPEC-0014-runtime-performance-contract.md).

## Kernel Identity

Production receipts must record the selected production QK256 CUDA kernel
identity and invocation counts. The status surface may display a user-friendly
name such as `qk256_gemv_cuda`, but the receipt must retain enough information
to distinguish production packed/scaled I2_S/QK256 kernels from diagnostic
F32/no-scale kernels.

Accepted production kernel identity fields include:

```text
selected_kernel_id
kernel_family = i2s_qk256
kernel_precision = scaled_i8s
kernel_backend = cuda
qk256_gemv_cuda_invocations
qk256_cuda_weight_uploads
```

If a receipt records CUDA linears but does not identify whether the kernel is
production packed/scaled I2_S/QK256, it cannot promote the BitNet proof
boolean.

## Strict Fallback Rules

For BitNet CUDA product, answer, benchmark, or server claims:

```text
fallback_used = false
fallback_reason = null
unsupported_strict_ops = 0
bitnet_linear_cpu_fallback_ops = 0
qk256_cuda_ops > 0
```

CPU fallback for tokenization, prompt rendering, sampling, or receipt
formatting may be allowed only when the receipt names it and the claim does not
depend on full residency. BitNet linear fallback is not allowed for strict
BitNet CUDA route proof.

## Accepted Proof Profiles

The initial governed BitNet CUDA profiles are:

```text
one_token
short_decode_8
short_decode_32
prefill_128_decode_16
prefill_512_decode_32
warm_session_3_turns
warm_session_10_turns
decode_128_from_warm_context
server_nonstream_chat_completions
```

One profile does not promote another. A one-token receipt does not prove short
decode, warm sessions, benchmark qualification, server readiness, speedup, or
full residency.

## Proof Boolean Criteria

`bitnet_packed_i2s_qk256_proof=true` is allowed only when the receipt proves
all of the following for the exact row and profile:

- artifact identity matches the supported I2_S/QK256 model row;
- tokenizer authority and prompt authority are recorded;
- `selected_backend="nvidia-rtx-5070-ti-cuda"`;
- `runtime_api="cuda"`;
- `selected_route="bitnet_qk256_cuda"`;
- production packed/scaled QK256 CUDA kernel identity is recorded;
- QK256 CUDA invocation count is greater than zero for the claimed execution
  profile;
- fallback is rejected;
- unsupported strict ops and BitNet linear CPU fallback counts are zero;
- answer quality or parity gate required by the profile passes;
- `dense_regular_llm_cuda_proof=false`.

The boolean must remain false for dense CUDA, generic CUDA, WGPU, Vulkan,
OpenCL, ROCm, Metal, CPU, hardware-only, diagnostic QK256, no-scale F32, or
planning-only receipts.

## Claim Boundaries

- Do not claim dense SLM CUDA from BitNet QK256 proof.
- Do not claim BitNet QK256 CUDA from dense SLM CUDA proof.
- Do not claim speedup without exact-profile benchmark review.
- Do not claim full residency from upload-once weights alone.
- Do not claim broad server readiness from server smoke or exact-profile
  non-streaming proof.
- Do not claim production packed I2_S/QK256 from diagnostic F32/no-scale
  QK256 parity.
- Do not claim strict RTX 5070 Ti CUDA from `selected_backend="cuda"`.

## Acceptance

This spec is satisfied when future BitNet CUDA PRs:

- record route, backend, runtime API, kernel identity, fallback, unsupported-op,
  and proof-family fields in receipts;
- keep dense proof booleans false for BitNet QK256 receipts;
- keep speed, residency, and broad server claims false until separate exact
  reviews accept them;
- update model status and receipt explanation without hand-editing generated
  dashboards;
- link hardware receipts or benchmark reports when promoting any claim.

## Proof Commands

Docs-only validation for this spec:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- check-model-coverage
```

Runtime PRs that promote BitNet CUDA claims must also include the exact command
that produced and explained the receipt, such as:

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- ask --device cuda --model <model> "..."
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- chat --device cuda --model <model>
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --model <model> --profile <profile>
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- receipts explain --latest --format json
```

## Non-Goals

- Do not implement kernels, routing, CLI, server, or benchmark behavior in this
  spec.
- Do not promote any model coverage row.
- Do not edit hardware proof receipts.
- Do not define dense regular-LLM CUDA proof.
- Do not define TL1, TL2, BF16, GPU-int2, OpenCL, ROCm, Metal, or NPU proof.
