# BITNET-SPEC-I2S-STATUS-SURFACE

Status: active
Owner: BitNet-rs maintainers
Created: 2026-05-19
Linked proposal: docs/proposals/BITNET-PROP-0015-i2s-productization.md
Linked specs:
  docs/specs/BITNET-SPEC-I2S-CUDA.md,
  docs/specs/BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md,
  docs/specs/BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md,
  docs/specs/BITNET-SPEC-SUPPORT-BUNDLE-SCHEMA.md,
  docs/specs/BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE.md
Linked ADRs: docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md
Linked plan: plans/i2s/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: yes
Policy impact: no

This spec defines the official BitNet I2_S/QK256 row that user-facing status,
receipt explanation, and support bundles must expose. It narrows the common
readiness/status surface to the current official BitNet CUDA product row.

## Purpose

The I2_S status surface must tell users what the official BitNet row has earned
without implying speed, residency, dense CUDA, or broad server readiness. The
row should be boring to trust: it shows the exact route, backend, proof family,
fallback state, accepted product tier, and next missing proof.

## Required Surfaces

These surfaces must agree for the official BitNet I2_S/QK256 row whenever they
can identify the same model coverage row:

```text
bitnet model status --device nvidia-rtx-5070-ti-cuda --format json
bitnet receipts explain <receipt> --format json
bitnet receipts explain --latest --format json
bitnet support bundle --latest --device nvidia-rtx-5070-ti-cuda --format json
```

Receipt explanation may report `null` for facts absent from an older receipt,
but it must not flip explicit unsupported claims to `true`.

## Canonical Row

The official BitNet I2_S/QK256 status row is:

```text
model_coverage_row = bitnet_official_2b_i2s_qk256
current_tier = product_cli_ready
tier = product_cli_ready
model_class = bitnet
artifact_kind = gguf_i2_s
tokenizer_authority = external_llama_bpe
prompt_authority = bitnetcpp-answer
selected_backend = nvidia-rtx-5070-ti-cuda
selected_route = bitnet_qk256_cuda
```

The row may display a friendlier model name, but the machine-readable row ID,
tier, backend, and route must remain stable unless the model coverage matrix is
intentionally migrated.

## Required Flat Fields

The I2_S row must preserve these flat fields across status, receipt
explanation, and support bundles:

```text
model_coverage_row
current_tier
tier
requested_backend
selected_backend
runtime_api
selected_route
fallback_used
product_cli_ready
server_ready
server_ready_scope
server_smoke
speedup_claim
full_residency_claim
benchmark_qualified
bitnet_packed_i2s_qk256_proof
dense_regular_llm_cuda_proof
claim_boundary
next_proof
```

`requested_backend` records user input or selector aliases. `selected_backend`
records the strict backend that status can support. `selected_route` records the
route that actually ran or the route the row is qualified for.

## Current Claim State

The current official BitNet I2_S/QK256 row is:

```text
product_cli_ready = true
cpu_answer_ready = true
accelerator_answer_ready = true
benchmark_qualified = false
server_ready = false
speedup_claim = false
full_residency_claim = false
bitnet_packed_i2s_qk256_proof = true
dense_regular_llm_cuda_proof = false
```

Strict server-smoke evidence may be displayed with `server_smoke=true` or
`server_ready_scope="smoke"`, but this must not become `server_ready=true`
unless a server-readiness review accepts a scoped profile. Current broad server
readiness is false.

## Unknown, False, And Smoke

The I2_S row must distinguish three states:

| Value | Meaning |
| --- | --- |
| `null` | The receipt or status source does not know this fact. |
| `false` | The repo has explicitly not accepted the claim. |
| `true` | The repo has accepted the claim for this exact row and scope. |

`server_smoke=true` is evidence that a bounded server smoke happened. It does
not imply exact-profile or broad server readiness. `benchmark_qualified=false`
with timing fields present means timing is reported but not accepted as a
qualified performance claim.

## Required Display Semantics

Human-readable status for the official I2_S row should communicate:

```text
Official BitNet 2B I2_S/QK256
  tier: product_cli_ready
  route: bitnet_qk256_cuda
  backend: nvidia-rtx-5070-ti-cuda
  ask/chat: ready
  server: smoke only; broad readiness false
  speedup: not qualified
  residency: not full-residency proven
  proof family: BitNet packed I2_S/QK256
  next proof: profile-specific speedup, residency timing, server readiness review
```

The exact wording may vary by renderer, but it must not hide false claim
booleans behind a generic "ready" label.

## Receipt And Support Bundle Joins

When a receipt joins to `bitnet_official_2b_i2s_qk256`, the explanation or
support bundle must preserve both:

- what happened in that receipt; and
- what the model coverage matrix currently allows the row to claim.

If a receipt lacks an execution plan, selected kernel, or fallback field, the
support bundle may still link it to the row, but it must warn that the receipt
is insufficient for route promotion. If the receipt is older than the current
claim boundary, it must not downgrade the row by itself; it remains evidence for
that run only.

## Claim Boundary Warnings

Status, receipt explanation, and support bundles should warn or preserve claim
limits when a user might infer any of these false claims:

- dense regular-LLM CUDA proof;
- Qwen2.5 or Qwen3 proof inheritance;
- generic CUDA proof;
- broad server readiness;
- speedup;
- full residency;
- WGPU, Vulkan, D3D12, OpenCL, ROCm, Metal, or CPU proof equivalence.

## Rejection Examples

| Status or receipt shape | Required response |
| --- | --- |
| `selected_route="dense_regular_llm_cuda"` with BitNet row | Preserve dense route fact and keep BitNet proof false for that receipt. |
| `selected_backend="cuda"` only | Do not present as strict RTX 5070 Ti CUDA proof. |
| `fallback_used=true` | Do not present as accelerator-answer-ready proof. |
| `server_smoke=true`, `server_ready=false` | Display smoke only; do not promote readiness. |
| `speedup_claim=true` without exact-profile review | Reject or warn; do not display accepted speedup. |
| Missing `next_proof` | Treat the row as incomplete for support automation. |

## Acceptance

This spec is accepted when:

- the row ID, tier, backend, route, and proof-family booleans are explicit;
- current official BitNet false claims remain false;
- server smoke is separated from server readiness;
- support surfaces are required to agree on backend, route, fallback, and claim
  booleans;
- no model coverage row, runtime behavior, or generated dashboard is changed.

## Proof Commands

Docs-only validation:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- check-model-coverage
```

If the same PR edits CLI status, receipt explanation, support bundles, or model
coverage rows, also run the focused tests listed in the common readiness and
receipt schema specs.

## Non-Goals

- Do not promote server readiness, speedup, full residency, or dense CUDA proof.
- Do not change receipt parsing or support-bundle code in this spec.
- Do not change model coverage rows, generated status dashboards, or hardware
  receipts.
- Do not use this status row as proof for non-official I2_S/TL1/TL2/GPU-int2
  candidates.
