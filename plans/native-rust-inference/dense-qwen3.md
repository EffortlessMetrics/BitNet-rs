# Dense Qwen3 Product Promotion

Qwen3 0.6B Q8_0 is product CLI-ready for bounded normal `ask` and `chat`
user paths on the RTX 5070 Ti dense CUDA route. It is not server-ready,
speed-qualified, benchmark-qualified, or full-residency-proven.

## Work item: CUDA-MODEL-009

Status: merged
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-MODEL-010
Blocked by: native inference plan

### Goal

Produce `docs/reports/CUDA_MODEL_009_QWEN3_PRODUCT_UX_AUDIT.md`.

### Production delta

No runtime delta. The audit maps `model verify`, ask, chat/warm path, receipt
explain, model status, fallback rejection, quality gate, benchmark review, and
claim booleans.

### Non-goals

No product promotion.

### Acceptance

Audit lists every user-path gap before Qwen3 can become product CLI-ready.

### Proof commands

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- check-model-coverage
```

### Rollback

Revert the audit report.

## Work item: CUDA-MODEL-010

Status: merged
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-MODEL-011
Blocked by: CUDA-MODEL-009

### Goal

Capture a strict Qwen3 ask user-path receipt.

### Production delta

The normal `bitnet ask` path produces valid decoded text with
`selected_backend=nvidia-rtx-5070-ti-cuda`, route `dense_regular_llm_cuda`, and
`fallback_used=false`.

### Non-goals

No speedup, server readiness, or product CLI promotion.

### Acceptance

Receipt explain works and `product_cli_ready` remains false unless review
promotes it.

### Proof commands

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- ask --device cuda --model <qwen3> "..."
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- receipts explain --latest --format json
```

### Rollback

Revert user-path changes and keep existing proof receipts unchanged.

## Work item: CUDA-MODEL-011

Status: merged
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-MODEL-012
Blocked by: CUDA-MODEL-010

### Goal

Capture Qwen3 chat or warm-session receipts.

### Production delta

Normal chat or warm-session path records model/tokenizer/context/weights loaded
once across multiple prompts.

### Non-goals

No server or speedup promotion.

### Acceptance

Session summary receipt shows `fallback_used=false`, `speedup=false`, and
`server=false`.

### Proof commands

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- chat --device cuda --model <qwen3>
```

### Rollback

Revert Qwen3 warm-session changes.

## Work item: CUDA-MODEL-012

Status: merged
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: Qwen3 server smoke/readiness review
Blocked by: none

### Goal

Review whether Qwen3 should be promoted to `product_cli_ready`.

### Production delta

Accepted for the bounded Qwen3 ask/chat CLI surface. Set Qwen3 product CLI
booleans while keeping server, speedup, benchmark-qualified, and full residency
false.

### Non-goals

No server-ready or speedup claim.

### Acceptance

Model coverage and status docs agree on the promoted tier and forbidden claims.
Qwen3 remains separate from Qwen2.5 and BitNet QK256 proof families.

### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```

### Rollback

Demote the row and revert status docs if future evidence invalidates the
accepted ask/chat user-path receipts.

## Work item: CUDA-MODEL-013

Status: merged
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-MODEL-014
Blocked by: CUDA-MODEL-012

### Goal

Record a current-source exact Qwen3 dense CUDA server-smoke receipt through the
shared-engine server path.

### Production delta

The server receipt path binds exact Qwen3 model identity to
`dense_qwen3_06b_q8_candidate` and `dense_regular_llm_cuda` for the RTX 5070 Ti
backend. A current-source receipt is committed at
`ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/server-strict-dense-qwen3-q8-smoke.json`.

### Non-goals

No Qwen3 server-ready, speedup, benchmark-qualified, full-residency, broad
dense GGUF, Qwen2.5-inheritance, or BitNet QK256 promotion.

### Acceptance

The Qwen3 dense server-smoke receipt uses the exact model ID/SHA, route,
backend, fallback, endpoint profile, generation policy, quality gate, and claim
booleans. Unknown dense model identities and wrong coverage rows are rejected by
the validator support landed earlier.

### Proof commands

```bash
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features qwen3_server_shared_engine
cargo test --locked -p bitnet-server --no-default-features --features cpu qwen3
```

### Rollback

Revert Qwen3 server receipt routing/validation support. Keep Qwen3 product CLI
coverage unchanged unless user-path evidence is invalidated separately.

## Work item: CUDA-MODEL-014

Status: merged
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: Qwen3 server-ready promotion
Blocked by: CUDA-MODEL-013

### Goal

Review whether Qwen3 has enough evidence for exact-profile server readiness.

### Production delta

Rejected at the time of review because no current-source Qwen3 non-streaming
`/v1/chat/completions` server-smoke receipt was committed. A later receipt now
exists at
`ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/server-strict-dense-qwen3-q8-smoke.json`,
and CUDA-MODEL-014B supersedes this rejected review for the exact-profile
promotion decision.

### Non-goals

No runtime change, model promotion, server-ready promotion, speedup,
benchmark-qualified, full-residency, broad dense GGUF, Qwen2.5-inheritance, or
BitNet QK256 claim.

### Acceptance

The review names the missing receipt, preserves all current false claim
booleans, and updates the Qwen3 next proof to require a committed current-source
server-smoke receipt before another readiness review.

### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```

### Rollback

Revert the review report and restore the previous Qwen3 next-proof text. No
runtime rollback is required.

## Work item: CUDA-MODEL-014B

Status: in_progress
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: Qwen3 server status UX
Blocked by: CUDA-MODEL-013

### Goal

Accept or reject Qwen3 exact-profile server readiness after reviewing the
committed current-source non-streaming shared-engine server-smoke receipt.

### Production delta

Accept exact-profile server readiness for `qwen3-0.6b-instruct-q8_0` only for
the Windows 9950X3D + RTX 5070 Ti `nvidia-rtx-5070-ti-cuda`
`dense_regular_llm_cuda` non-streaming `/v1/chat/completions` profile recorded
in
`ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/server-strict-dense-qwen3-q8-smoke.json`.

### Non-goals

No runtime change, speedup, benchmark-qualified speed, full-residency, broad
dense GGUF, Qwen2.5-inheritance, or BitNet QK256 claim.

### Acceptance

The model coverage row promotes `server_ready=true` only for the exact Qwen3
server profile, preserves speed/residency/benchmark/BitNet false claims, keeps
the runtime receipt's `server_ready_claimed=false`, and updates model-status and
receipt-explain contract tests to show `server_ready_scope=exact_profile`.

### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- check-model-coverage
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli model_status_dashboard_lists_qwen3_as_product_cli_ready
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli receipts_explain_links_qwen3_dense_receipt_to_product_cli_coverage
git diff --check
```

### Rollback

Restore the previous Qwen3 model coverage `server_ready=false` state and remove
the CUDA-MODEL-014B acceptance report. No runtime rollback is required.
