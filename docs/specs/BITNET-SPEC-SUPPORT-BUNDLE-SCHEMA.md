# BITNET-SPEC-SUPPORT-BUNDLE-SCHEMA

Status: proposed
Linked proposal:
[BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
Linked specs:
[BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE](BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md),
[BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA](BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md)
Linked ADRs:
[BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Applies to: `bitnet support bundle --latest --device <device> --format json`,
GitHub CUDA support issues, receipt triage docs

## Purpose

The support bundle is the pasteable issue artifact for proof-carrying local
inference. It combines model status, latest receipt explanation, binary/build
identity, and runtime identity into one JSON object without running inference
or promoting claims.

This spec defines the support bundle schema and support semantics.

## Source-Of-Truth Authorities

The support bundle composes existing surfaces:

- `bitnet model status --device <device> --format json`;
- `bitnet receipts explain <receipt> --format json`;
- `ci/model-artifacts/model-coverage-matrix.toml`;
- [Model readiness/status surface](BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md);
- [Receipt explain schema](BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md);
- [CUDA receipt triage guide](../tutorials/cuda-receipt-triage.md);
- `.github/ISSUE_TEMPLATE/cuda-support.yml`.

The bundle is a support envelope. It is not a receipt and is not proof by
itself.

## Top-Level JSON Contract

`bitnet support bundle --latest --device <device> --format json` must emit an
object with at least these fields:

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

`kind` must identify the object as a support bundle. `created_utc` records when
the bundle was assembled. `device` records the requested device label.

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
receipt_path
```

`summary` must be derived from `latest_receipt` and the corresponding
`model_status` row when possible. It must not use stale docs prose or free-form
string matching when the structured surfaces provide a value.

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

Unknown build fields may be `null`. They must not be replaced by placeholder
values that look authoritative.

## Runtime Identity

`runtime` should include runtime identity when the latest receipt contains it:

```text
selected_backend
runtime_api
device_name
driver_version
cuda_runtime_version
cuda_driver_version
source
```

Runtime identity may be `null` on CPU-only or older receipts. A missing CUDA
driver/runtime field must not be interpreted as CPU fallback, CUDA failure, or
successful CUDA proof.

## Embedded Status And Receipt Objects

`model_status` must preserve the full model status dashboard shape from
[BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE](BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md).

`latest_receipt` must preserve the full receipt explanation shape from
[BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA](BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md).

The bundle may add fields in future versions, but it must not remove these
embedded objects without a schema-version bump.

## No New Inference

Support bundle generation must be read-only:

- it may read the model coverage matrix;
- it may read a receipt path or resolve `--latest`;
- it may inspect build/runtime identity already available locally;
- it must not download models;
- it must not run ask, chat, bench, serve, or hardware probes;
- it must not create a new inference receipt.

This keeps support collection cheap and safe for users filing issues.

## Proof-Family And Claim Boundaries

The bundle must preserve the same hard boundaries as status and receipt
explanation:

- `speedup_claim=false` remains false unless an exact-profile benchmark review
  accepted it.
- `full_residency_claim=false` remains false unless per-phase residency proof
  accepted it.
- Exact-profile server readiness must show its scope.
- Dense CUDA proof must not satisfy BitNet packed I2_S/QK256 proof.
- BitNet QK256 proof must not satisfy dense SLM proof.
- Qwen2.5 proof must not satisfy Qwen3 proof.

## Issue Template Contract

CUDA support issue templates should ask for the support bundle JSON before
free-form environment prose. If the command fails, the template should ask for
the failed command, stderr, and any available receipt path.

## Proof Commands

Current contract validation:

```bash
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli support_bundle
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli receipts_explain
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli model_status_dashboard
git diff --check
```

## Non-Goals

- Do not make the support bundle an immutable receipt.
- Do not run inference, model verification, benchmarks, server requests, or
  hardware probes while assembling a bundle.
- Do not promote any model, server, speedup, residency, or proof-family claim.
- Do not require CUDA-only runtime fields for non-CUDA support bundles.

## Related Policy Or Manifest Sources

- `ci/model-artifacts/model-coverage-matrix.toml`
- `docs/tutorials/cuda-receipt-triage.md`
- `.github/ISSUE_TEMPLATE/cuda-support.yml`
- `docs/tracking/campaigns/nvidia-5070ti/active.toml`
