# Server Readiness And Product UX

## Work item: CUDA-UX-011

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`,
`docs/specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: server display, support triage, automation dashboards
Blocked by: native inference plan

### Goal

Make `bitnet model status --device nvidia-rtx-5070-ti-cuda --format json` and
`bitnet receipts explain --latest --format json` stable enough for automation.

### Production delta

Machine-readable status exposes model coverage row, current tier, selected
backend, selected route, fallback status, server scope, speedup, residency,
BitNet proof, and dense CUDA proof.

### Non-goals

No runtime route change, model promotion, speedup claim, or server-ready
promotion.

### Acceptance

JSON includes at least:

```json
{
  "model_coverage_row": "dense_qwen25_05b_q8_cuda",
  "current_tier": "product_cli_ready",
  "selected_backend": "nvidia-rtx-5070-ti-cuda",
  "selected_route": "dense_regular_llm_cuda",
  "fallback_used": false,
  "server_ready": true,
  "speedup_claim": false,
  "full_residency_claim": false,
  "bitnet_packed_i2s_qk256_proof": false,
  "dense_regular_llm_cuda_proof": true
}
```

### Proof commands

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- model status --device nvidia-rtx-5070-ti-cuda --format json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- receipts explain --latest --format json
```

### Rollback

Remove the JSON additions and keep text status behavior unchanged.

## Work item: CUDA-UX-012

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: support issue templates
Blocked by: CUDA-UX-011 optional

### Goal

Add `docs/tutorials/cuda-receipt-triage.md` so users and maintainers can debug
receipt failures without reading model coverage TOML.

### Production delta

Support docs explain `fallback_used=true`, generic CUDA selection, tokenizer
authority gaps, prompt template gaps, quality failures, false speed/server
claims, and dense-vs-BitNet proof mistakes.

### Non-goals

No CLI behavior change and no claim promotion.

### Acceptance

Guide includes what to paste into a GitHub issue and the forbidden-claim
boundaries for dense CUDA versus BitNet proof.

### Proof commands

```bash
git diff --check
```

### Rollback

Revert the tutorial.

## Work item: SERVER-READY-001

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: per-request receipt export
Blocked by: CUDA-UX-011

### Goal

Display server readiness scope plainly in model status.

### Production delta

Qwen2.5 shows exact-profile readiness for non-streaming
`/v1/chat/completions`; official BitNet shows server smoke evidence while broad
`server_ready` remains false.

### Non-goals

No new server promotion.

### Acceptance

Status distinguishes `server_ready=true scope=exact_profile` from
`server_smoke=true server_ready=false`.

### Proof commands

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- model status --device nvidia-rtx-5070-ti-cuda --format json
```

### Rollback

Revert the display changes without touching receipts.

## Work item: SERVER-READY-002

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: server support workflows
Blocked by: SERVER-READY-001

### Goal

Export per-request receipts from server paths.

### Production delta

Expose `/receipts/latest`, `/receipts/{id}`, `/readiness`, and `/v1/models`
without adding a second inference engine.

### Non-goals

No new `server_ready` promotion.

### Acceptance

Response metadata includes a receipt ID, and the readiness row links to the
model coverage row.

### Proof commands

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- serve --device cuda --model <model>
```

### Rollback

Remove receipt export endpoints and retain existing server behavior.
