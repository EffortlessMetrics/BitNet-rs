# BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE

Status: proposed
Linked proposal:
[BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
Linked specs:
[BITNET-SPEC-0001](BITNET-SPEC-0001-source-of-truth-and-claim-boundaries.md),
[BITNET-SPEC-0013](BITNET-SPEC-0013-model-onboarding-proof-ladder.md),
[BITNET-SPEC-CUDA-ROUTE-CONTRACT](BITNET-SPEC-CUDA-ROUTE-CONTRACT.md),
[BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE](BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md),
[BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA](BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md)
Linked ADRs:
[BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Applies to: model coverage rows, route contracts, backend receipts, model
status JSON, receipt explanation JSON, support bundles, support docs, PR claim
text, campaign promotion reviews

## Purpose

BitNet-rs supports multiple model families, backend APIs, route IDs, hardware
profiles, and proof surfaces. Evidence from one lane is useful only if it does
not silently promote another lane.

This spec defines the common non-inheritance contract: proof does not transfer
across model family, artifact, tokenizer/prompt authority, backend, route,
hardware profile, server profile, benchmark profile, or residency class unless
a narrower accepted spec explicitly says so and a promotion PR records the
exact bridge.

## Source-Of-Truth Authorities

Non-inheritance is enforced by these authorities:

- `ci/model-artifacts/model-coverage-matrix.toml`;
- route contracts such as
  [CUDA Route Contract](BITNET-SPEC-CUDA-ROUTE-CONTRACT.md);
- [Model Onboarding Proof Ladder](BITNET-SPEC-0013-model-onboarding-proof-ladder.md);
- [Model Readiness Status Surface](BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md);
- [Receipt Explain Schema](BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md);
- immutable receipts under `ci/hardware/**`;
- backend, model-family, performance, server, and residency specs for the
  exact lane being promoted.

Docs prose, issue text, PR titles, branch names, generated dashboards, and
campaign tracker summaries may describe proof. They do not create inheritance.

## Identity Dimensions

A proof claim is scoped by all applicable identity dimensions:

```text
model family
model artifact and checksum
tokenizer authority
prompt-template authority
quantization or layout
backend API
selected backend
selected route
hardware profile
runtime profile
server endpoint/profile
benchmark profile
fallback status
quality gate
receipt or comparator identity
```

A promotion may reuse prior evidence only when every required identity
dimension is the same, or when a narrower accepted spec defines the allowed
substitution and its acceptance evidence.

## Required Fields

Any status row, receipt explanation, support bundle, or promotion review that
summarizes backend/model proof must preserve these fields when known:

```text
model_coverage_row
model_id or artifact id
current_tier
requested_backend
selected_backend
runtime_api
selected_route
fallback_used
quality_gate
server_ready
server_ready_scope
speedup_claim
full_residency_claim
bitnet_packed_i2s_qk256_proof
dense_regular_llm_cuda_proof
next_proof
```

Unknown facts must remain `null` or explicitly unknown. A missing field must
not be treated as `false` evidence of strict execution, and it must never be
treated as `true` support.

## Hard Non-Inheritance Rules

These boundaries are mandatory:

- Dense CUDA proof is not BitNet packed I2_S/QK256 CUDA proof.
- BitNet packed I2_S/QK256 CUDA proof is not dense SLM CUDA proof.
- Qwen2.5 proof is not Qwen3 proof.
- Qwen3 proof is not Qwen2.5, SmolLM2, Llama, Gemma, Phi, or BitNet proof.
- SmolLM2 structural validity is not SmolLM2 CPU answer readiness.
- One dense SLM row does not prove another dense SLM row.
- BitNet I2_S/QK256 proof is not TL1, TL2, GPU-int2, BF16, or diagnostic
  no-scale QK256 proof.
- OpenVINO GPU proof is not native OpenCL proof.
- OpenVINO NPU proof is not Arc GPU proof.
- A770 OpenCL proof is not Intel Lunar Lake OpenVINO proof.
- Apple CPU/NEON proof is not Metal, ANE, CUDA, OpenCL, ROCm, or NPU proof.
- CUDA proof is not OpenCL, ROCm, Metal, OpenVINO, NPU, WGPU, Vulkan, D3D12, or
  CPU proof.
- CPU fallback is not selected accelerator proof.
- Hardware detection is not selected backend execution proof.
- Route visibility is not selected route execution proof.
- Diagnostic trace parity is not product answer quality.
- Layer-planning evidence is not whole-model execution.
- One-token proof is not chat, warm-session, benchmark, or server readiness.
- CLI ready is not server ready.
- Server smoke is not broad server readiness.
- Non-streaming server proof is not streaming proof.
- Single-request server proof is not concurrency, uptime, deployment, or broad
  production readiness.
- Benchmark evidence is not speedup unless the exact benchmark profile is
  accepted.
- Speed benchmark evidence is not residency proof.
- Upload-once, kernel-count, or transfer evidence is not full residency unless
  every required residency phase is proven.

## Promotion Requirements

A promotion PR that crosses a readiness tier, server boundary, speed boundary,
or proof-family boolean must:

1. name the exact model coverage row;
2. name the exact artifact, tokenizer authority, and prompt authority;
3. name the requested backend, selected backend, runtime API, and selected
   route;
4. cite the exact receipt, comparator, or gate evidence;
5. show `fallback_used=false` for selected-route accelerator claims;
6. preserve unrelated proof-family booleans as `false`;
7. state which adjacent claims remain false or unknown;
8. point claim text back to this spec or a narrower route/model spec.

If any field is unavailable, keep the broader claim false and record the next
proof instead.

## Acceptance Examples

| Evidence | May claim | Must not claim |
| --- | --- | --- |
| Qwen2.5 dense CUDA ask/chat receipt | exact Qwen2.5 dense CUDA CLI path | Qwen3, SmolLM2, BitNet QK256, broad dense GGUF, speedup, or full residency |
| Qwen3 dense CUDA ask/chat receipt | exact Qwen3 dense CUDA CLI path | Qwen2.5, server readiness, speedup, full residency, or BitNet proof |
| Official BitNet QK256 CUDA receipt | exact official BitNet packed I2_S/QK256 CUDA path | dense SLM CUDA, TL1/TL2, GPU-int2, broad server readiness, or speedup |
| A770 OpenCL diagnostic trace | diagnostic visibility or trace parity for the exact probe | product quality, selected RTX 5070 Ti proof, speed, or readiness |
| OpenVINO NPU static proof | scoped OpenVINO NPU static-subgraph evidence | Arc GPU, native OpenCL, CUDA, full inference, or speedup |
| Server smoke receipt | one bounded endpoint/profile response | broad server readiness, streaming, concurrency, speedup, or residency |
| Microbenchmark | exact kernel/profile measurement | end-to-end product speedup or model readiness |

## Surface Alignment

When a claim-bearing PR updates any one of these surfaces, it must preserve the
same non-inheritance semantics across the rest of the touched support surface:

```text
ci/model-artifacts/model-coverage-matrix.toml
bitnet model status --device <device> --format json
bitnet receipts explain <receipt> --format json
bitnet support bundle --latest --device <device> --format json
docs/status/*
docs/tutorials/*
```

Status, receipt explanation, and support bundles may summarize a claim. They
must not create a claim that is absent from the model coverage row or receipt.

## Proof Commands

Docs-only validation for this spec:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- check-model-coverage
```

PRs that change route behavior, receipts, model coverage, server readiness,
performance, or residency must also run the proof commands required by the
exact route/model/performance spec they touch.

## Non-Goals

- Do not promote any current model, backend, server, speed, residency, or proof
  claim in this spec.
- Do not encode today's PR queue, branch names, or campaign order.
- Do not define route-specific kernels, tokenizers, prompt templates, or
  benchmark thresholds.
- Do not replace backend-specific, model-family, performance, server, or
  residency specs.
- Do not make support bundles, tutorials, generated dashboards, or PR prose a
  proof source.

## Related Policy Or Manifest Sources

- `ci/model-artifacts/model-coverage-matrix.toml`
- `ci/model-artifacts/artifact-manifest.toml`
- `ci/model-artifacts/tokenizer-authority.toml`
- `docs/status/CUDA_CAPABILITY_MATRIX.md`
- `ci/hardware/**`
- `docs/tracking/campaigns/**`
- `policy/docs-source-of-truth.toml`
