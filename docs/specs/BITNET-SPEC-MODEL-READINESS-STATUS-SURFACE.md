# BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE

Status: proposed
Linked proposal:
[BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
Linked specs:
[BITNET-SPEC-0013](BITNET-SPEC-0013-model-onboarding-proof-ladder.md),
[BITNET-SPEC-0014](BITNET-SPEC-0014-runtime-performance-contract.md),
[BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE](BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE.md)
Linked ADRs:
[BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Applies to: `bitnet model status --format json`, receipt explanation
summaries, support bundles, model coverage rows, status dashboards

## Purpose

`bitnet model status` is the front door for verified local inference. It must
show what is supported, what route would run, what backend is selected, which
proof family applies, and which claims remain false. It must not collapse model
loading, answer quality, CUDA execution, performance, and server readiness into
one vague "works" state.

This spec defines the stable readiness/status surface that user-facing status,
receipt explanation, and support bundles must preserve.

## Source-Of-Truth Authorities

The status surface summarizes existing authorities:

- `ci/model-artifacts/model-coverage-matrix.toml`;
- [Model Coverage Matrix](../model-artifacts/MODEL_COVERAGE_MATRIX.md);
- [Model Onboarding Proof Ladder](BITNET-SPEC-0013-model-onboarding-proof-ladder.md);
- [Runtime Performance Contract](BITNET-SPEC-0014-runtime-performance-contract.md);
- [Proof-family non-inheritance](BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE.md);
- [Proof-family ADR](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md);
- `ci/hardware/**` receipts;
- campaign active manifests and events under `docs/tracking/campaigns/**`.

If a status row disagrees with the model coverage matrix, repair the status row
or the matrix before promoting the user-facing claim. If a status row disagrees
with a receipt, the receipt remains evidence for what happened and the status
row remains the supported claim state.

## Top-Level JSON Contract

`bitnet model status --device <device> --format json` must emit a stable object
with at least these fields:

```text
schema_version
device
requested_backend
selected_backend
source
note
models
```

`schema_version` changes only when a breaking shape change is intentionally
accepted. `requested_backend` records the user's input. `selected_backend`
records the strict backend label that status rows use for proof, or `null` when
the device cannot be mapped to a supported status surface.

## Model Row Contract

Each row in `models` must include the support and claim fields required for
automated support:

```text
id
model_coverage_row
display_name
model_class
route
selected_route
requested_backend
selected_backend
tier
current_tier
status
category
fallback_used
cpu_answer_ready
accelerator_answer_ready
benchmark_qualified
product_cli_ready
speedup_claim
server_ready
full_residency_claim
bitnet_packed_i2s_qk256_proof
dense_regular_llm_cuda_proof
ask
one_token
short_decode
warm_session
benchmark
server
server_ready_scope
server_scope
server_endpoint
server_streaming
server_smoke
server_reason
claim_boundary
next_proof
```

`tier` and `current_tier` are aliases for the same model coverage tier. New
fields may be added, but existing fields must not be removed or repurposed
without a schema-version bump and migration note.

## Tier And Readiness Semantics

Readiness booleans are not interchangeable:

- `cpu_answer_ready=true` means the CPU answer gate passed for that row.
- `accelerator_answer_ready=true` means the exact accelerator/backend/route
  proof passed without hidden fallback.
- `product_cli_ready=true` means normal user CLI paths are accepted for the row.
- `benchmark_qualified=true` means a benchmark review accepted the exact
  profile. It does not imply global speedup.
- `server_ready=true` means the server profile in the row was accepted. It does
  not imply streaming, concurrency, or broad readiness unless the row says so.

`category` is a display grouping, not a proof tier. `status` is a user-facing
summary, not an authority that overrides `current_tier` or claim booleans.

## Unknown, False, And Smoke Semantics

The status surface must distinguish unknown from false:

- `null` means the row has no selected route, fallback result, or readiness
  scope to report.
- `false` means the repo has explicitly not accepted that claim.
- `server_smoke=true` means bounded smoke evidence exists.
- `server_ready=false` with `server_smoke=true` means smoke evidence is not
  broad readiness.

Status renderers must not translate `null` to `false` when the distinction
matters for support, and must not translate smoke evidence to readiness.

## Proof-Family Boundaries

The status surface must preserve proof-family booleans:

- `bitnet_packed_i2s_qk256_proof=true` applies only to the exact BitNet
  packed I2_S/QK256 row and route.
- `dense_regular_llm_cuda_proof=true` applies only to the exact dense model row
  and route.
- A dense proof must never satisfy a BitNet QK256 proof.
- A BitNet QK256 proof must never satisfy a dense SLM proof.
- Qwen2.5 proof must never satisfy Qwen3, SmolLM2, Llama, Gemma, or Phi rows.

Any row that is product CLI-ready or server-ready must still show unrelated
proof-family booleans as `false`.

## Required Parity Across Surfaces

The following surfaces must agree for the same model coverage row:

```text
bitnet model status --device <device> --format json
bitnet receipts explain <receipt> --format json
bitnet support bundle --latest --device <device> --format json
```

They must preserve:

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

Receipt explanation may report `null` when the receipt lacks information; model
status remains the current claim state from the matrix.

## Proof Commands

Current contract validation:

```bash
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli model_status_dashboard
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli receipts_explain
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```

## Non-Goals

- Do not promote any model tier or claim in this spec.
- Do not change runtime math, kernels, tokenizer, loader, server behavior, or
  model coverage rows.
- Do not make `server_smoke` imply `server_ready`.
- Do not make one proof family imply another proof family.
- Do not require model downloads or hardware runs to validate this docs-only
  contract.

## Related Policy Or Manifest Sources

- `ci/model-artifacts/model-coverage-matrix.toml`
- `docs/status/CUDA_CAPABILITY_MATRIX.md`
- `docs/tracking/campaigns/nvidia-5070ti/active.toml`
- `docs/tracking/generated/*`
- `policy/docs-source-of-truth.toml`
