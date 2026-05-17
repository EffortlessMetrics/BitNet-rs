# CUDA Server Readiness

This plan page sequences server readiness work for the 9950X3D + RTX 5070 Ti
CUDA product lane. It implements the boundary in
[BITNET-SPEC-0010](../../docs/specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md)
without promoting any model by itself.

## Source-Of-Truth Links

- Proposal:
  [`BITNET-PROP-0002`](../../docs/proposals/BITNET-PROP-0002-9950x3d-5070ti-cuda-productization.md)
- CUDA product contract:
  [`BITNET-SPEC-0007`](../../docs/specs/BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md)
- Server readiness boundary:
  [`BITNET-SPEC-0010`](../../docs/specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md)
- CUDA campaign:
  [`docs/tracking/campaigns/nvidia-5070ti/CAMPAIGN.md`](../../docs/tracking/campaigns/nvidia-5070ti/CAMPAIGN.md)
- Model coverage:
  `ci/model-artifacts/model-coverage-matrix.toml`
- Receipt root:
  `ci/hardware/windows-9950x3d-rtx5070ti/**`

## Current State

Dense Qwen2.5 0.5B Q8_0 has a bounded strict CUDA server-smoke receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/server-strict-dense-qwen25-q8-smoke.json
```

That receipt is evidence for the bounded smoke path. It is not, by itself, a
`server_ready=true` promotion. CUDA-SERVER-003 audited it against
BITNET-SPEC-0010 and found the receipt is missing artifact checksum identity,
endpoint or request-profile scope, and generation-policy fields. CUDA-SERVER-004
hardens future server receipts to emit those fields and adds a receipt validator,
but still does not promote `server_ready=true`. CUDA-SERVER-005 refreshes the
dense Qwen receipt from the hardened server path:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-17/server-strict-dense-qwen25-q8-smoke.json
```

That receipt carries the artifact SHA-256, endpoint/request profile, generation
policy, strict backend, dense route, quality gate, and false speed/full
residency/BitNet proof booleans required by BITNET-SPEC-0010. The model
coverage row promotes `server_ready=true` only for that exact bounded profile.

Official BitNet I2_S/QK256 does not have a server-readiness claim from the dense
Qwen server smoke. It needs its own exact-profile server receipt before any
server row can promote.

## Work Item: CUDA-SERVER-003

Status: merged
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007, BITNET-SPEC-0010
Linked ADRs: BITNET-ADR-0004
Campaign item: `CUDA-SERVER-003`
Blocked by: CUDA-SERVER-002
Blocks: exact-profile server readiness promotions

### Goal

Apply the server readiness promotion checklist to the bounded dense Qwen2.5
server-smoke evidence before changing any model coverage boolean.

### Production Delta

Docs and status alignment only. CUDA-SERVER-003 records that the current server
smoke is not promotable as-is.

### Non-Goals

No broad production serving claim, no BitNet server claim from dense Qwen, no
speedup, no full-residency claim, and no default PR CI expansion.

### Acceptance

- The exact model coverage row is identified.
- The exact server receipt path is identified.
- Missing artifact checksum, endpoint/profile scope, and generation-policy
  fields are recorded as blockers.
- `server_ready=true` remains false.
- `speedup_claim=false` and `full_residency_claim=false` remain unchanged.

### Proof Commands

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --locked -p xtask --no-default-features -- campaign generate --check
```

### Rollback

Revert the docs or model coverage promotion from the PR. Historical server
smoke receipts stay immutable evidence for what happened.

## Work Item: CUDA-SERVER-004

Status: merged
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007, BITNET-SPEC-0010
Linked ADRs: BITNET-ADR-0004
Campaign item: `CUDA-SERVER-004`
Blocked by: CUDA-SERVER-003
Blocks: refreshed dense Qwen server-smoke receipt and exact-profile promotion

### Goal

Harden the shared-engine server receipt shape so future dense Qwen2.5 server
smoke receipts carry artifact checksum identity, endpoint or request-profile
scope, generation policy, and claim booleans required by BITNET-SPEC-0010.

### Production Delta

Runtime receipts and `receipts explain` become stricter and more useful for
support triage. The existing historical server-smoke receipt remains unchanged
and should warn as missing the newly required exact-profile fields.

### Non-Goals

No committed refreshed hardware receipt, no `server_ready=true` promotion, no
global dense server readiness, no BitNet QK256 server proof, no speedup, and no
full-residency claim.

### Acceptance

- Future server shared-engine receipts include model identity and SHA-256 when
  available.
- Dense Qwen route, model-coverage row, server-smoke, and dense CUDA inference
  claim emission require the exact `qwen2.5-0.5b-instruct-q8_0` request model
  plus matching active artifact SHA-256 loaded through CUDA; missing or
  mismatched SHA, or a CPU-loaded active model, stays on a generic shared-engine
  receipt with no dense route or coverage claim.
- Future receipts include endpoint/request profile and generation-policy scope.
- A server shared-engine validator rejects generic CUDA, fallback, missing
  exact-profile fields, premature server-ready claims, speed claims, and BitNet
  proof conflation.
- `receipts explain` warns on stale server-smoke receipts that lack the new
  fields.
- `server_ready=false` remains unchanged in model coverage.

### Proof Commands

```bash
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features server_shared_engine_chat_completion
cargo test --locked -p bitnet-server --no-default-features --features cpu server_shared_engine_receipt
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,full-cli receipts_explain
git diff --check
```

### Claim Boundary

This cannot claim global dense GGUF server readiness, BitNet QK256 server
readiness, speedup, full CUDA residency, concurrency, or production deployment
readiness.

## Work Item: CUDA-SERVER-005

Status: in_progress
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007, BITNET-SPEC-0010
Linked ADRs: BITNET-ADR-0004
Campaign item: `CUDA-SERVER-005`
Blocked by: refreshed dense Qwen server shared-engine receipt with exact-profile fields and validator pass
Blocks: dense Qwen server status UX

### Goal

Promote dense Qwen2.5 server readiness only for the exact bounded profile after
a refreshed or supplemental receipt carries the artifact checksum, endpoint or
request-profile scope, and generation policy required by BITNET-SPEC-0010 and
passes the server shared-engine receipt validator.

### Production Delta

The dense Qwen2.5 0.5B Q8_0 model coverage row becomes server-ready only for
the refreshed non-streaming RTX 5070 Ti shared-engine `/v1/chat/completions`
profile. The promotion does not change runtime behavior.

### Acceptance

- The refreshed receipt path is
  `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-17/server-strict-dense-qwen25-q8-smoke.json`.
- The receipt includes model SHA-256, endpoint profile, request profile,
  generation policy, strict selected backend, runtime API, dense route, quality
  gate, and false speed/full-residency/BitNet proof booleans.
- The receipt passes the `server_shared_engine_chat_completion` validator.
- `ci/model-artifacts/model-coverage-matrix.toml`,
  `docs/model-artifacts/MODEL_COVERAGE_MATRIX.md`, and
  `docs/status/CUDA_CAPABILITY_MATRIX.md` agree on the exact-profile
  `server_ready=true` promotion.
- Speedup, full CUDA residency, global dense GGUF server readiness, BitNet
  QK256 server readiness, concurrency, deployment readiness, and broad chat
  quality remain false/non-claims.

### Proof Commands

```bash
python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-17\server-strict-dense-qwen25-q8-smoke.json
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features server_shared_engine_chat_completion
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-17\server-strict-dense-qwen25-q8-smoke.json
cargo run --locked -p xtask --no-default-features -- check-model-coverage
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --locked -p xtask --no-default-features -- campaign generate --check
cargo run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports --fail-on-error
git diff --check
```

### Claim Boundary

This cannot claim global dense GGUF server readiness, BitNet QK256 server
readiness, speedup, full CUDA residency, concurrency, or production deployment
readiness.

## Work Item: CUDA-SERVER-006

Status: proposed
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007, BITNET-SPEC-0010
Linked ADRs: BITNET-ADR-0004
Campaign item: `CUDA-SERVER-006`
Blocked by: CUDA-SERVER-004
Blocks: official BitNet server status UX

### Goal

Add or promote official BitNet I2_S/QK256 strict server smoke separately from
dense Qwen.

### Claim Boundary

Official BitNet server proof must use `route = bitnet_qk256_cuda`, preserve
QK256 invocation evidence, keep dense regular-LLM proof false, and keep speed
false unless a separate benchmark qualification accepts an exact profile.
