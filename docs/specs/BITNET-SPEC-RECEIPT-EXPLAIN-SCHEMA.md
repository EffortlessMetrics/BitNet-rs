# BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA

Status: proposed
Linked proposal:
[BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
Linked specs:
[BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE](BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md),
[BITNET-SPEC-0013](BITNET-SPEC-0013-model-onboarding-proof-ladder.md),
[BITNET-SPEC-0014](BITNET-SPEC-0014-runtime-performance-contract.md),
[BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE](BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE.md)
Linked ADRs:
[BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Applies to: `bitnet receipts explain <receipt> --format json`,
`bitnet receipts explain --latest --format json`, receipt support summaries

## Purpose

`bitnet receipts explain` turns a raw receipt into a stable support summary. It
must explain what happened in the run and link that run to the current model
coverage row without inventing unsupported claims.

This spec defines the normalized JSON schema for receipt explanation.

## Source-Of-Truth Authorities

Receipt explanation summarizes:

- the receipt JSON selected by the command;
- `ci/model-artifacts/model-coverage-matrix.toml`;
- [Model readiness/status surface](BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md);
- [Runtime Performance Contract](BITNET-SPEC-0014-runtime-performance-contract.md);
- [Proof-family non-inheritance](BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE.md);
- [Proof-family ADR](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md).

The raw receipt is evidence for what happened. The model coverage row is the
current support claim. Receipt explanation must show both when they differ.

## Top-Level JSON Contract

`bitnet receipts explain <receipt> --format json` must emit an object with at
least these fields:

```text
schema_version
path
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
artifact_kind
claim
model
model_coverage
backend
execution_plan
kernels
quality
timing
residency
benchmark_qualification
openvino
claim_limits
```

The flat aliases exist for support automation. The nested objects preserve
diagnostic detail.

## Latest Resolution

With `--latest`, the command may search the default receipt directory or a
provided directory for the newest JSON receipt. The returned `path` must be the
receipt that was actually explained. `--latest` must not run inference, produce
a new receipt, or silently select non-JSON artifacts.

If latest resolution fails, the command must fail with an actionable path or
receipt-discovery message rather than returning an empty success object.

## Flat Alias Semantics

The flat aliases must be filled from the best available source:

1. explicit receipt fields;
2. normalized receipt subobjects such as `backend`, `execution_plan`, or
   `claim_boundary`;
3. the linked model coverage row;
4. `null` when unknown.

They must not default unknown claims to `true`. A missing receipt field may be
`null`; an explicitly unaccepted model-coverage claim remains `false`.

## Required Nested Objects

The nested objects must preserve these responsibilities:

| Object | Owns |
| --- | --- |
| `model_coverage` | row, tier, route, product/server/speed/residency/proof booleans, warnings |
| `backend` | requested backend, selected backend, runtime API, fallback status/reason |
| `execution_plan` | selected route, model family, profile, route-specific claims |
| `quality` | quality gate pass/fail, blocker, answer/UTF-8 status when present |
| `timing` | phase timing and throughput fields when present |
| `residency` | residency, transfer, VRAM, and workspace reuse evidence when present |
| `benchmark_qualification` | accepted/rejected/not-reviewed benchmark decision |
| `claim_limits` | forbidden claims and promotion warnings |

Receipts from older lanes may leave some nested fields empty, but the object
shape must remain stable.

## Proof-Family And Promotion Warnings

Receipt explanation must warn when users might infer the wrong proof family:

- BitNet QK256 proof is not dense SLM CUDA proof.
- Dense regular-LLM CUDA proof is not BitNet packed I2_S/QK256 proof.
- Qwen2.5 proof is not Qwen3 proof.
- Server smoke is not broad server readiness.
- Benchmark evidence is not speedup unless the exact profile was accepted.

Warnings may be nested under `model_coverage.warnings` or `claim_limits`, but
the JSON must keep proof-family booleans explicit at the top level.

## Unknown, False, And Conflict Semantics

Receipt explanation must preserve three states:

- `null`: the receipt or matrix did not provide this fact.
- `false`: the claim is explicitly not accepted.
- `true`: the claim is explicitly accepted for the row or receipt scope.

If the raw receipt says a claim was not made but the model coverage row has an
accepted exact-profile claim, explanation may show the accepted row claim, but
it must preserve the receipt's raw claim boundary in nested detail or warnings.

## Required Rows For Contract Tests

Schema contract tests must cover at least:

- official BitNet 2B I2_S/QK256;
- Qwen2.5 0.5B Q8_0 dense CUDA;
- Qwen3 0.6B Q8_0 dense CUDA;
- SmolLM2 360M structurally valid blocker row.

The tests must assert `model_coverage_row`, `current_tier`,
`selected_backend`, `selected_route`, `fallback_used`, `server_ready`,
`server_ready_scope`, `speedup_claim`, `full_residency_claim`,
`bitnet_packed_i2s_qk256_proof`, and `dense_regular_llm_cuda_proof`.

## Proof Commands

Current contract validation:

```bash
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli receipts_explain
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```

## Non-Goals

- Do not require receipt explanation to run inference or benchmarks.
- Do not promote model, server, speedup, residency, or proof-family claims.
- Do not make the receipt schema a replacement for raw immutable receipts.
- Do not hide conflicts between raw receipt claims and current model coverage.

## Related Policy Or Manifest Sources

- `ci/model-artifacts/model-coverage-matrix.toml`
- `ci/hardware/**`
- `docs/tracking/campaigns/**`
- `policy/docs-source-of-truth.toml`
