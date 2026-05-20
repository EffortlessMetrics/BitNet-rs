# BITNET-SPEC-I2S-STATUS-SURFACE: I2_S/QK256 Status Contract

Status: active
Owner: BitNet-rs maintainers
Created: 2026-05-19
Linked proposal: [BITNET-PROP-0015](../proposals/BITNET-PROP-0015-i2s-productization.md)
Linked specs:
[BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE](BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md),
[BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA](BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md),
[BITNET-SPEC-SUPPORT-BUNDLE](BITNET-SPEC-SUPPORT-BUNDLE.md),
[BITNET-SPEC-CUDA-ROUTE-CONTRACT](BITNET-SPEC-CUDA-ROUTE-CONTRACT.md),
[BITNET-SPEC-I2S-CUDA](BITNET-SPEC-I2S-CUDA.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [I2_S implementation plan](../../plans/i2s/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines how I2_S/QK256 support state appears in model
  status, receipt explanation, and support bundles.
Policy impact: no

## Purpose

This spec defines the user-facing I2_S/QK256 status surface. It keeps the
official BitNet packed QK256 route readable without implying dense SLM proof,
speedup, full residency, or broad server readiness.

## Scope

The status contract applies to:

```text
bitnet model status --device nvidia-rtx-5070-ti-cuda --format json
bitnet receipts explain <receipt> --format json
bitnet support bundle --latest --device nvidia-rtx-5070-ti-cuda --format json
```

It summarizes the current claim state. It does not replace the model coverage
matrix or a receipt for what actually happened in one run.

## Hard Rule

I2_S/QK256 status rows must show exact artifact, tier, selected backend,
selected route, fallback status, proof-family booleans, speed/residency/server
claim booleans, and the next missing proof. Status text must not turn smoke,
diagnostic, dense, or CPU evidence into production BitNet CUDA support.

## Required Row Fields

An I2_S/QK256 status row that is shown in model status, receipt explanation, or
support bundles must preserve these fields when known:

```text
model_coverage_row
current_tier
product_cli_ready
requested_backend
selected_backend
runtime_api
selected_route
selected_kernel_id
fallback_used
unsupported_strict_ops
bitnet_linear_cpu_fallback_ops
server_ready
server_ready_scope
server_smoke
speedup_claim
benchmark_qualified
full_residency_claim
bitnet_packed_i2s_qk256_proof
dense_regular_llm_cuda_proof
quality_gate
next_proof
forbidden_claims
```

Receipts may report `null` or `unknown` for fields absent from older receipts.
Model status remains the current supported claim state from the coverage row.

## Official BitNet 2B RTX 5070 Ti Display Contract

The bounded product row for the official Microsoft BitNet 2B I2_S/QK256 CUDA
lane must render with these claim boundaries until separate receipts promote
them:

```text
current_tier = product_cli_ready
product_cli_ready = true
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
selected_route = bitnet_qk256_cuda
fallback_used = false when a strict CUDA receipt is being explained
bitnet_packed_i2s_qk256_proof = true
dense_regular_llm_cuda_proof = false
speedup_claim = false
full_residency_claim = false
server_ready = false unless a server-readiness review promotes the exact scope
```

If bounded server smoke exists, the status surface may expose it as
`server_smoke=true` or a human-readable note. It must not translate smoke into
`server_ready=true` or broad readiness.

## Surface Agreement

For the same model row and latest receipt, the three user-support surfaces must
agree on:

```text
model_coverage_row
current_tier
selected_backend
selected_route
fallback_used
product_cli_ready
server_ready
server_ready_scope
speedup_claim
full_residency_claim
bitnet_packed_i2s_qk256_proof
dense_regular_llm_cuda_proof
next_proof
```

When a receipt lacks a field needed for agreement, `receipts explain` must say
that the field is unknown or missing. It must not infer a positive claim.

## Forbidden Status Inferences

Status renderers, receipt explainers, support bundles, generated dashboards,
and docs must not infer:

- dense SLM CUDA proof from BitNet QK256 proof;
- BitNet QK256 proof from dense SLM CUDA proof;
- broad server readiness from server smoke;
- server readiness from product CLI readiness;
- speedup from answer readiness;
- full residency from upload-once weights;
- strict RTX 5070 Ti proof from generic `cuda`;
- production packed I2_S/QK256 proof from diagnostic F32/no-scale QK256;
- Qwen2.5 or Qwen3 proof from the official BitNet row.

## Next-Proof Semantics

`next_proof` must name the next missing evidence rather than a generic "todo".
Common I2_S/QK256 values include:

```text
repeated profile benchmark review
TTFT/residency bottleneck audit
exact-profile server readiness review
full-residency phase proof
kernel identity receipt upgrade
fallback-free warm-session proof
```

A status row may list multiple blockers, but the first item should be the next
actionable proof needed for the lane.

## Acceptance

This status contract is accepted when:

- model status JSON exposes the fields above for the official BitNet row;
- receipt explanation preserves route/backend/fallback/proof-family fields for
  BitNet CUDA receipts;
- support bundles embed the same claim booleans without promotion;
- generated status docs continue to show speedup and full residency as false
  until exact-profile reviews accept them;
- server smoke stays visible without becoming broad server readiness.

## Proof Commands

Current contract validation:

```bash
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli model_status_dashboard
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli receipts_explain
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```

Docs-only PRs may run the model-coverage and diff checks and record CLI test
gaps when the local environment cannot build the CLI.

## Non-Goals

- Do not change model coverage rows in this spec.
- Do not hand-edit generated dashboards.
- Do not promote server readiness, speedup, or full residency.
- Do not change runtime math, kernels, tokenizer, loader, or server behavior.
