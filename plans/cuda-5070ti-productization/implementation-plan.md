# CUDA 5070 Ti Productization Implementation Plan

This queue starts after the source-of-truth proposal, product contract spec, and
product-bench ADR. Each item is intentionally PR-sized and keeps claim
promotion tied to receipts.

## Queue

| Order | Work item | PR title | Primary file |
| --- | --- | --- | --- |
| 1 | CUDA-PROD-008 | `docs(cuda): reconcile 5070 Ti BitNet and dense proof state` | campaign and status docs |
| 2 | CUDA-PROD-009 | `cuda(bitnet): harden strict ask/chat user preflight` | CLI/product path |
| 3 | CUDA-PROD-010 | `cuda(bitnet): benchmark qualification receipts for official I2_S` | benchmark receipts |
| 4 | CUDA-UX-009 | `docs(cuda): update BitNet user guide for strict CUDA` | tutorial |
| 5 | CUDA-DENSE-050 | `cuda(dense): audit Qwen2.5 Q8_0 proof state` | dense audit report |
| 6 | CUDA-DENSE-051 | `cuda(dense): implement or refresh Qwen one-token strict CUDA proof` | dense receipt path |
| 7 | CUDA-DENSE-052 | `cuda(dense): Qwen short decode strict CUDA proof` | dense receipt path |
| 8 | CUDA-DENSE-053 | `cuda(dense): Qwen warm-session chat proof` | dense receipt path |
| 9 | CUDA-DENSE-054 | `cuda(dense): Qwen benchmark qualification` | benchmark receipts |
| 10 | CUDA-MODEL-001 | `model(cuda): add Qwen3 0.6B artifact contract` | model artifact docs |
| 11 | CUDA-MODEL-002 | `model(cuda): add Qwen3 CPU answer sanity` | CPU receipts |
| 12 | CUDA-MODEL-003 | `model(cuda): add Qwen3 CUDA all-layer plan` | all-layer plan |
| 13 | CUDA-MODEL-004 | `model(cuda): add Qwen3 one-token CUDA proof` | CUDA receipt path |
| 14 | CUDA-MODEL-005 | `model(cuda): add Qwen3 short-decode and warm-session proof` | CUDA receipt path |
| 15 | CUDA-UX-008 | `cli(cuda): model support dashboard` | CLI/status surface |
| 16 | CUDA-UX-010 | `docs(cuda): 9950X3D+5070Ti CUDA quickstart` | tutorial |
| 17 | CUDA-SERVER-001 | `server(cuda): strict CUDA server smoke` | server path |

## Shared Links

All work items link to:

- Proposal:
  `docs/proposals/BITNET-PROP-0002-9950x3d-5070ti-cuda-productization.md`
- Spec:
  `docs/specs/BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md`
- ADR:
  `docs/adr/BITNET-ADR-0004-9950x3d-5070ti-cuda-product-bench.md`
- Campaign:
  `docs/tracking/campaigns/nvidia-5070ti/active.toml`
- Model coverage:
  `ci/model-artifacts/model-coverage-matrix.toml`
- Receipt root:
  `ci/hardware/windows-9950x3d-rtx5070ti/**`

## Work item: CUDA-PROD-008

Status: ready
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007, `rtx5070ti-cuda-answer-readiness`
Linked ADRs: BITNET-ADR-0004
Campaign item: `CUDA-PROD-008`
Blocked by: merged proposal, spec, ADR
Blocks: CUDA-PROD-009, CUDA-DENSE-050

### Goal

Reconcile campaign, model coverage, status, and plan docs so there is one
current-state ledger for official BitNet, dense Qwen, and candidate models.

### Production delta

Docs only. No new receipt or claim promotion.

### Non-goals

No code, model manifest, receipt, workflow, generated-dashboard, or README
product-claim change.

### Acceptance

Add a table with current state, last real receipt, next missing proof, allowed
claim, and forbidden claim for:

- official BitNet 2B I2_S CUDA;
- dense Qwen2.5 0.5B Q8_0 CUDA;
- Qwen3 0.6B;
- SmolLM2 360M;
- Llama 3.2 1B and 3B;
- Gemma/Phi small.

### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Receipt paths

Use committed receipts under:

```text
ci/hardware/windows-9950x3d-rtx5070ti/**
```

### Claim boundary

No new model, CUDA, answer, speed, server, or residency claim.

### Rollback

Revert the docs-only reconciliation. Receipts and ledgers stay unchanged.

## Work item: CUDA-PROD-009

Status: planned
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007, `rtx5070ti-cuda-answer-readiness`
Linked ADRs: BITNET-ADR-0004
Campaign item: `CUDA-PROD-009`
Blocked by: CUDA-PROD-008
Blocks: CUDA-PROD-010, CUDA-UX-009

### Goal

Make strict BitNet CUDA user commands fail closed before generation and print a
compact proof summary when they can write a receipt.

### Production delta

Harden normal `bitnet cuda doctor`, `bitnet model verify`, `bitnet ask`, and
warm-session paths for the official BitNet I2_S/QK256 artifact.

### Non-goals

No dense Qwen work, no benchmark speed promotion, no server path.

### Acceptance

- Missing tokenizer fails before generation.
- Generic `cuda` does not silently become RTX 5070 Ti proof.
- CPU fallback fails under strict CUDA.
- Default receipt path and compact summary are visible.
- `speedup_claim=false` remains default.

### Proof commands

```bash
cargo test --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli cuda_doctor ask_strict
cargo check --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
git diff --check
```

### Receipt paths

```text
target/bitnet/receipts/cuda-answer-readiness/strict-cuda-ask-latest.json
ci/hardware/windows-9950x3d-rtx5070ti/**
```

### Claim boundary

Strict preflight and summary behavior only. No new answer-quality or speed
claim unless receipts already prove the exact case.

### Rollback

Revert CLI preflight and summary changes; existing receipts remain evidence for
prior runs.

## Work item: CUDA-PROD-010

Status: planned
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007
Linked ADRs: BITNET-ADR-0004
Campaign item: `CUDA-PROD-010`
Blocked by: CUDA-PROD-008
Blocks: CUDA-UX-009

### Goal

Make official BitNet I2_S/QK256 speed decisions governed and profile-specific.

### Production delta

Add benchmark qualification receipts for `one_token`, `short_decode_8`,
`short_decode_32`, `warm_session_3_turns`, and `warm_session_10_turns`.

### Non-goals

No global CUDA speedup claim. No dense Qwen speed claim.

### Acceptance

Each profile records CPU and CUDA p50/p95/mean, prefill, first token, steady
decode, kernel time, H2D/D2H timing source, VRAM high-water mark, thermal or
power context when available, fallback status, decision, and reason.

### Proof commands

```bash
cargo test --locked -p bitnet-bench-receipts --no-default-features
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --cuda-benchmark-receipt <receipt>
git diff --check
```

### Receipt paths

```text
ci/hardware/windows-9950x3d-rtx5070ti/<date>/bitnet-i2s-<profile>-benchmark.json
```

### Claim boundary

`speedup_claim=true` may apply only to an accepted exact profile and model.

### Rollback

Revert generated benchmark qualification docs or receipts from the PR. Do not
edit historical receipts by hand.

## Work items: CUDA-DENSE-050 through CUDA-DENSE-054

Status: planned
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007
Linked ADRs: BITNET-ADR-0004
Campaign items: `CUDA-DENSE-050` through `CUDA-DENSE-054`
Blocked by: CUDA-PROD-008
Blocks: CUDA-MODEL-001

### Goal

Audit and then productize dense Qwen2.5 0.5B Q8_0 as the first dense CUDA SLM
lane without inheriting any BitNet QK256 claim.

### Production delta

See [`dense-qwen.md`](dense-qwen.md).

### Non-goals

No BitNet proof, no generic dense GGUF claim, no server readiness.

### Acceptance

The lane must distinguish real hardware receipts from validators/contracts and
then add or refresh one-token, short-decode, warm-session, and benchmark
receipts as needed.

### Proof commands

```bash
python -m json.tool <receipt>
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
git diff --check
```

### Receipt paths

```text
ci/hardware/windows-9950x3d-rtx5070ti/<date>/dense-qwen25-q8-*.json
```

### Claim boundary

`dense_regular_llm_cuda_proof` may become true only for exact committed Qwen
receipts. `bitnet_packed_i2s_qk256_proof=false` stays explicit.

### Rollback

Revert the dense-lane PR and demote any status row if the receipt no longer
proves the claim.

## Work items: CUDA-MODEL-001 through CUDA-MODEL-005

Status: planned
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007
Linked ADRs: BITNET-ADR-0004
Campaign items: `CUDA-MODEL-001` through `CUDA-MODEL-005`
Blocked by: CUDA-DENSE-050
Blocks: later SmolLM2/Llama/Gemma/Phi candidate ladders

### Goal

Use Qwen3 0.6B as the first test of generalized dense model onboarding.

### Production delta

See [`small-llm-candidates.md`](small-llm-candidates.md).

### Non-goals

Do not batch-promote all candidates. Do not inherit Qwen2.5 evidence.

### Acceptance

Artifact contract, CPU sanity, all-layer plan, one-token CUDA, and short-decode
or warm-session proof land as separate PRs.

### Proof commands

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
```

### Receipt paths

```text
ci/model-artifacts/<model-id>.toml
ci/hardware/windows-9950x3d-rtx5070ti/<date>/<model>-*.json
```

### Claim boundary

Candidate rows stay candidate until their own proof ladders pass.

### Rollback

Revert only the candidate row or receipt introduced by the failed PR.

## Work items: CUDA-UX-008, CUDA-UX-010, CUDA-SERVER-001

Status: planned
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007
Linked ADRs: BITNET-ADR-0004
Campaign items: `CUDA-UX-008`, `CUDA-UX-010`, `CUDA-SERVER-001`
Blocked by: BitNet and dense Qwen CLI proof surfaces
Blocks: broader product docs

### Goal

Expose the proof state through user-facing status, quickstart, and later server
smoke paths.

### Production delta

- `bitnet model status --device nvidia-rtx-5070-ti-cuda`
- `docs/tutorials/9950x3d-5070ti-cuda-quickstart.md`
- strict CUDA server smoke only after CLI is stable

### Non-goals

No server production-readiness claim before a fallback-free server receipt.

### Acceptance

Status and quickstart commands say what each row proves and does not prove.
Server readiness stays false until `CUDA-SERVER-001` lands.

### Proof commands

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- model status --device nvidia-rtx-5070-ti-cuda
git diff --check
```

### Receipt paths

```text
ci/model-artifacts/model-coverage-matrix.toml
ci/hardware/windows-9950x3d-rtx5070ti/<date>/server-strict-cuda-smoke.json
```

### Claim boundary

Status and docs summarize proof. They do not create new proof.

### Rollback

Revert the status/docs/server-smoke PR and demote the server row if needed.
