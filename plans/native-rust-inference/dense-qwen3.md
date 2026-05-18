# Dense Qwen3 Product Promotion

Qwen3 0.6B Q8_0 is accelerator-answer-ready candidate evidence. It is not yet
product CLI-ready, server-ready, speed-qualified, or full-residency-proven.

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

Status: in_progress
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: Qwen3 server readiness review
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
