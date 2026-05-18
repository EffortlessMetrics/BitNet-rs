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
| 14 | CUDA-MODEL-005 | `model(cuda): add Qwen3 short-decode CUDA proof` | CUDA receipt path |
| 15 | CUDA-MODEL-006 | `model(cuda): add Qwen3 warm-session CUDA proof` | CUDA receipt path |
| 16 | CUDA-MODEL-007 | `model(cuda): add Qwen3 benchmark qualification review` | benchmark receipts |
| 17 | CUDA-MODEL-008 | `model(cuda): sync Qwen3 earned status row` | model coverage and status |
| 18 | CUDA-UX-008 | `cli(cuda): model support dashboard` | CLI/status surface |
| 19 | CUDA-UX-010 | `docs(cuda): 9950X3D+5070Ti CUDA quickstart` | tutorial |
| 20 | CUDA-SERVER-001 | `server(cuda): strict CUDA server smoke` | server path |
| 21 | CUDA-SERVER-002 | `server(cuda): commit dense Qwen strict smoke receipt` | server receipt path |
| 22 | CUDA-MODEL-SMOLLM2-001 | `model(cuda): add SmolLM2 360M artifact contract` | model artifact docs |
| 23 | CUDA-MODEL-SMOLLM2-002 | `docs(cuda): sync SmolLM2 CPU blocker state` | model coverage and plan docs |
| 24 | CUDA-SERVER-003 | `docs(cuda): define server readiness promotion boundary` | server readiness spec |
| 25 | CUDA-SERVER-004 | `server(cuda): harden dense Qwen server receipt fields` | server receipt path |
| 26 | CUDA-SERVER-005 | `docs(cuda): promote dense Qwen exact-profile server readiness` | model coverage and status |
| 27 | CUDA-SERVER-006 | `server(cuda): official BitNet strict server smoke` | server receipt path |

## Shared Links

All work items link to:

- Proposal:
  `docs/proposals/BITNET-PROP-0002-9950x3d-5070ti-cuda-productization.md`
- Spec:
  `docs/specs/BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md`
- Server readiness spec:
  `docs/specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md`
- ADR:
  `docs/adr/BITNET-ADR-0004-9950x3d-5070ti-cuda-product-bench.md`
- Campaign:
  `docs/tracking/campaigns/nvidia-5070ti/active.toml`
- Model coverage:
  `ci/model-artifacts/model-coverage-matrix.toml`
- Receipt root:
  `ci/hardware/windows-9950x3d-rtx5070ti/**`

## Current-State Ledger

| Lane | Current state | Last real receipt | Next missing proof |
| --- | --- | --- | --- |
| BitNet official 2B I2_S CUDA | product CLI ready, speed false | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-bitnet-perf-003-warm-session-benchmark.json` | profile-specific benchmark qualification |
| Dense Qwen2.5 0.5B Q8_0 CUDA | product CLI ready in model coverage; real strict runtime receipts, benchmark qualification reviews, and bounded server-smoke receipts exist; speed and broad server readiness stay false | `docs/reports/CUDA_SERVER_003_DENSE_QWEN_READINESS_AUDIT.md` | refresh or supplement server-smoke evidence with artifact checksum identity, endpoint/profile scope, and generation policy before any exact-profile `server_ready=true` row |
| Qwen3 0.6B | accelerator-ready dense SLM candidate; one-token, short-decode, warm-session, and benchmark-review evidence exists; product CLI, speed, server, full residency, broad dense GGUF, and BitNet proof stay false | `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-benchmark-qualification.json` | user-facing ask/chat product UX or repeated same-artifact comparator evidence before any product CLI or speed profile promotion |
| SmolLM2 360M | structurally valid artifact contract; strict CPU preflight blocked before tokenizer/prompt/generation; governed normalization-policy audit recorded | `ci/slm-cpu/windows-9950x3d-rtx5070ti/2026-05-16/smollm2-360m-normalization-policy-audit.json` | implement exact metadata-scoped SmolLM2 normalization validation and retry CPU sanity before all-layer planning or CUDA |
| Llama 3.2 1B | registered candidate | none | artifact contract, tokenizer/prompt authority, CPU sanity |
| Llama 3.2 3B | registered candidate | none | memory envelope, artifact contract, tokenizer/prompt authority |
| Gemma/Phi small | registered candidate | none | architecture policy, artifact contract, tokenizer/prompt authority |

This ledger does not promote claims. It records the reconciliation target for
CUDA-PROD-008 and the audit target for CUDA-DENSE-050.

## Work item: CUDA-PROD-008

Status: merged
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

Status: merged
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

Status: merged
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

Status: merged
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

Status: merged through CUDA-MODEL-008 and CUDA-MODEL-SMOLLM2-001; SmolLM2 CPU sanity is blocked by SLM-CPU-017 strict-loader evidence
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007
Linked ADRs: BITNET-ADR-0004
Campaign items: `CUDA-MODEL-001` through `CUDA-MODEL-008`,
`CUDA-MODEL-SMOLLM2-001`, `CUDA-MODEL-SMOLLM2-002`
Blocked by: CUDA-DENSE-050
Blocks: later SmolLM2/Llama/Gemma/Phi candidate ladders

### Goal

Use Qwen3 0.6B as the first test of generalized dense model onboarding.

### Production delta

See [`small-llm-candidates.md`](small-llm-candidates.md).

### Non-goals

Do not batch-promote all candidates. Do not inherit Qwen2.5 evidence.

### Acceptance

Qwen3 artifact contract, CPU sanity, all-layer plan, one-token CUDA,
short-decode, warm-session, benchmark review, and earned status sync landed as
separate PRs. SmolLM2 360M artifact-contract proof has landed, and SLM-CPU-017
records a strict CPU preflight blocker before tokenizer/prompt/generation. The
next candidate proof is an SLM CPU loader-policy follow-up before any SmolLM2
CPU answer, CUDA, product CLI, speed, server, full-residency, or BitNet claim.

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

## Work item: CUDA-MODEL-SMOLLM2-001

Status: merged
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007
Linked ADRs: BITNET-ADR-0004
Campaign item: `CUDA-MODEL-SMOLLM2-001`
Blocked by: CUDA-MODEL-008
Blocks: SmolLM2 CPU sanity and CUDA route planning

### Goal

Start the next dense SLM candidate after Qwen3 by adding an exact SmolLM2 360M
artifact contract.

### Production delta

Add or complete the SmolLM2 360M model artifact contract and report with source,
file identity, checksum, GGUF metadata, tokenizer and prompt authority, license,
context length, memory envelope, and current claim state.

### Non-goals

No CPU answer readiness, CUDA proof, product CLI readiness, speedup, server
readiness, full CUDA residency, broad dense GGUF support, or BitNet QK256 proof.

### Acceptance

- Exact source/repository and file identity are recorded.
- SHA256, byte size, GGUF type, architecture, quantization, tokenizer, chat
  template, context length, license, storage envelope, and VRAM estimate are
  recorded when available.
- `ci/model-artifacts/model-coverage-matrix.toml` remains candidate-only unless
  the artifact contract proves a narrower tier.
- The row keeps `cpu_answer_ready=false`, `accelerator_answer_ready=false`,
  `product_cli_ready=false`, `server_ready=false`, `speedup_claim=false`, and
  `bitnet_packed_i2s_qk256_proof=false`.

### Proof commands

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
```

### Receipt paths

```text
ci/model-artifacts/<smollm2-360m-model-id>.toml
docs/reports/SMOLLM2_360M_ARTIFACT_CONTRACT.md
```

### Claim boundary

This work can only claim that SmolLM2 360M has an artifact contract or remains a
registered candidate with an identified next proof. It cannot claim answer
readiness or CUDA execution.

### Rollback

Revert the SmolLM2 artifact contract/report and restore the model coverage row
to registered candidate state.

## Work item: CUDA-MODEL-SMOLLM2-002

Status: merged
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007
Linked ADRs: BITNET-ADR-0004
Campaign item: `CUDA-MODEL-SMOLLM2-002`
Blocked by: CUDA-MODEL-SMOLLM2-001, SLM-CPU-017
Blocks: SmolLM2 wrong-first-token diagnosis, dense all-layer planning, and CUDA route planning

### Goal

Sync the CUDA productization and model coverage surfaces to the committed
SmolLM2 strict CPU blocker chain.

### Production delta

Docs and model coverage only. The model row records that SmolLM2 360M reached
strict CPU model-load preflight on the 9950X3D, passed exact metadata-scoped
normalization validation in a later SLM CPU item, and then reached one-token
generation with `fallback_used=false` but selected the wrong first token.

### Non-goals

No runtime code, loader-policy change, CPU answer claim, CUDA route claim,
product CLI readiness, speedup, server readiness, full-residency claim, broad
dense GGUF claim, or BitNet QK256 proof.

### Acceptance

- `ci/model-artifacts/model-coverage-matrix.toml` records the SmolLM2 row as
  structurally valid with CPU answer readiness blocked by wrong-first-token
  diagnosis.
- The row links the next proof to a reference-compatible first-token/top-k or
  checkpoint comparator capture using the SLM-CPU-022 contract.
- The NVIDIA campaign and CUDA productization plan point to the SLM CPU blocker
  and diagnosis receipts as the last real evidence.
- The SmolLM2 ladder does not start CUDA planning until CPU sanity is
  unblocked.

### Proof commands

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
```

### Receipt paths

```text
ci/slm-cpu/windows-9950x3d-rtx5070ti/2026-05-15/smollm2-360m-strict-cpu-preflight-blocker.json
ci/slm-cpu/windows-9950x3d-rtx5070ti/2026-05-16/smollm2-360m-strict-cpu-sanity-retry.json
ci/slm-cpu/windows-9950x3d-rtx5070ti/2026-05-16/smollm2-360m-wrong-first-token-diagnosis.json
ci/slm-cpu/windows-9950x3d-rtx5070ti/2026-05-16/smollm2-360m-reference-comparator-contract.json
```

### Claim boundary

This work can only claim that the SmolLM2 artifact contract exists and that a
strict CPU quality blocker is recorded through wrong-first-token diagnosis and
comparator contract. It cannot claim CPU answer readiness, CUDA execution,
product CLI readiness, speed, server readiness, full residency, broad dense GGUF
support, or BitNet QK256 proof.

### Rollback

Revert the SmolLM2 blocker-status wording and restore the post-artifact-contract
candidate text.

## Work items: CUDA-UX-008, CUDA-UX-010, CUDA-SERVER-001, CUDA-SERVER-002

Status: merged through CUDA-SERVER-002
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007
Linked ADRs: BITNET-ADR-0004
Campaign items: `CUDA-UX-008`, `CUDA-UX-010`, `CUDA-SERVER-001`,
`CUDA-SERVER-002`
Blocked by: BitNet and dense Qwen CLI proof surfaces
Blocks: broader product docs

### Goal

Expose the proof state through user-facing status, quickstart, and later server
smoke paths.

### Production delta

- `bitnet model status --device nvidia-rtx-5070-ti-cuda`
- `docs/tutorials/9950x3d-5070ti-cuda-quickstart.md`
- bounded dense Qwen strict CUDA server-smoke receipt

### Non-goals

No broad server production-readiness claim from a bounded server-smoke receipt.

### Acceptance

Status and quickstart commands say what each row proves and does not prove. The
dense Qwen server-smoke receipt exists, but `server_ready` remains false until a
later exact-profile readiness promotion spec permits it.

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

## Work items: CUDA-SERVER-003 through CUDA-SERVER-006

Status: CUDA-SERVER-003 and CUDA-SERVER-004 merged; CUDA-SERVER-005 in progress; CUDA-SERVER-006 proposed
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007, BITNET-SPEC-0010
Linked ADRs: BITNET-ADR-0004
Campaign items: `CUDA-SERVER-003` through `CUDA-SERVER-006`
Blocked by: CUDA-SERVER-002
Blocks: exact-profile server readiness status promotion

### Goal

Define and then apply exact-profile server readiness promotion without turning
bounded smoke evidence into broad server support.

### Production delta

See [`server-readiness.md`](server-readiness.md).

### Non-goals

No global server readiness, no speedup, no full-residency claim, and no
cross-family proof inheritance.

### Acceptance

`CUDA-SERVER-003` audited the bounded dense Qwen server-smoke receipt against
the readiness boundary and recorded that it is not promotable as-is.
`CUDA-SERVER-004` adds the missing shared-engine receipt fields and validator
for future receipts without promoting `server_ready`. Later promotion PRs can
set `server_ready=true` only for the exact model/profile whose refreshed or
supplemental receipt satisfies the server readiness spec and passes the server
shared-engine receipt validator.
`CUDA-SERVER-004` also gates dense Qwen server route and claim emission on the
exact request model plus active artifact SHA-256 loaded through CUDA; missing or
mismatched artifact identity, or a CPU-loaded active model, stays generic and
cannot populate dense coverage or server-smoke claims.
`CUDA-SERVER-005` refreshes the dense Qwen shared-engine server receipt from
that hardened path and promotes `server_ready=true` only for the exact
non-streaming RTX 5070 Ti `/v1/chat/completions` profile. Speedup, full
residency, broad dense server readiness, production deployment readiness, and
BitNet QK256 proof remain false/non-claims.

### Proof commands

```bash
git diff --check
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features server_shared_engine_chat_completion
cargo test --locked -p bitnet-server --no-default-features --features cpu server_shared_engine_receipt
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
```

### Receipt paths

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/server-strict-dense-qwen25-q8-smoke.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-17/server-strict-dense-qwen25-q8-smoke.json
ci/hardware/windows-9950x3d-rtx5070ti/<date>/server-strict-bitnet-i2s-qk256-smoke.json
```

### Claim boundary

Server readiness is exact-profile only and cannot imply BitNet/dense
cross-family proof, speedup, full residency, concurrency, or production
deployment readiness.

### Rollback

Revert the docs or promotion row introduced by the PR. Do not edit historical
server receipts by hand.
