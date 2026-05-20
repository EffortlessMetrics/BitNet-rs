# BITNET-SPEC-SUPPORT-BUNDLE

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-19
Linked proposal:
[BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
Linked specs:
[BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE](BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md),
[BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA](BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md),
[BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE](BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE.md),
[BITNET-SPEC-CUDA-SUPPORT-ISSUE](BITNET-SPEC-CUDA-SUPPORT-ISSUE.md)
Linked ADRs:
[BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: support bundles expose existing claim boundaries only
Policy impact: none

## Purpose

`bitnet support bundle --latest --device <device> --format json` is the
pasteable support artifact for local inference issues. It lets a user or agent
answer from one JSON object:

```text
Can I run this model?
On which backend?
What is proven?
What is explicitly not proven?
What command generated the proof?
What should I paste into an issue if it fails?
What is the next blocker?
```

The bundle is a support envelope. It is not a receipt and does not run
inference.

## Source-Of-Truth Authorities

The support bundle composes these structured sources:

- `bitnet model status --device <device> --format json`;
- `bitnet receipts explain <receipt> --format json`;
- `ci/model-artifacts/model-coverage-matrix.toml`;
- the latest local receipt selected by `--latest` or by an explicit path;
- build identity available in the compiled binary;
- runtime identity already present in the receipt.

The bundle must not derive claims from stale docs prose, free-form issue text,
or filename heuristics when structured status or receipt fields exist.

## Top-Level JSON Contract

The JSON object must include:

```text
schema_version
kind
created_utc
device
summary
binary
runtime
model_status
latest_receipt
```

`kind` must be `bitnet_support_bundle`. `schema_version` is currently `1`.
Adding optional fields is compatible with schema version `1`; removing or
renaming fields requires a schema-version bump.

## Summary Contract

`summary` is the issue-triage front panel. It must include:

```text
model_coverage_row
current_tier
selected_backend
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
claim_boundary
receipt_path
```

`summary.claim_boundary` must carry the structured model-status or receipt
claim boundary when available. If no structured boundary is available, it may be
`null`; it must not be replaced by a broad support claim.

## Binary Identity

`binary` must include:

```text
name
crate_version
git_commit
git_commit_source
build_timestamp
rustc_version
target_triple
```

Unknown build fields may be `null`. They must not be filled with placeholder
values that look authoritative.

## Runtime Identity

`runtime` should include runtime identity already present in the receipt:

```text
selected_backend
runtime_api
device_name
driver_version
cuda_runtime_version
cuda_driver_version
source
```

Missing runtime identity means unknown. It must not be interpreted as CPU
fallback, CUDA failure, or successful CUDA proof.

## Embedded Status And Receipt Objects

`model_status` must preserve the full dashboard shape defined by
[BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE](BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md).

`latest_receipt` must preserve the full explanation shape defined by
[BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA](BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md).

The embedded objects are intentionally redundant with `summary`. `summary`
helps issue triage; embedded objects preserve audit detail.

## No New Inference

Support bundle generation is read-only:

- it may read the model coverage matrix;
- it may resolve or read a receipt path;
- it may inspect build identity already compiled into the binary;
- it must not download models;
- it must not run `ask`, `chat`, `bench`, `serve`, or hardware probes;
- it must not create a new inference receipt.

## Claim Boundaries

The bundle must preserve these boundaries:

- diagnostic evidence is not semantic quality;
- route visibility is not selected execution;
- Qwen2.5 proof is not Qwen3 proof;
- dense CUDA proof is not BitNet packed I2_S/QK256 proof;
- one exact-profile server receipt is not broad server readiness;
- microbench evidence is not product speedup;
- missing fallback data is unknown, not false proof of strict execution.

## CUDA Issue Contract

CUDA issue templates must ask for support bundle JSON before free-form
environment prose. See
[BITNET-SPEC-CUDA-SUPPORT-ISSUE](BITNET-SPEC-CUDA-SUPPORT-ISSUE.md).

If bundle generation fails, the issue template or triage path should ask for:

```text
failed command
stderr
receipt path, if available
GPU / driver / runtime version, if the bundle did not capture it
```

## Proof Commands

Current contract validation:

```bash
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli support_bundle
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli cuda_support_issue_template
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli receipts_explain
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli model_status_dashboard
git diff --check
```

## Non-Goals

- Do not make the support bundle an immutable receipt.
- Do not run inference, model verification, benchmarks, server requests, or
  hardware probes while assembling a bundle.
- Do not promote model, server, speedup, residency, or proof-family claims.
- Do not require CUDA-only runtime fields for non-CUDA support bundles.

## Compatibility Alias

[BITNET-SPEC-SUPPORT-BUNDLE-SCHEMA](BITNET-SPEC-SUPPORT-BUNDLE-SCHEMA.md)
is a stable compatibility URL for older references. This file owns the current
support-bundle contract.
