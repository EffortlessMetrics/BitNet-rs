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
endpoint or request-profile scope, and generation-policy fields. The model
coverage row remains false until a later promotion PR supplies those fields and
applies the exact-profile requirements in BITNET-SPEC-0010.

Official BitNet I2_S/QK256 does not have a server-readiness claim from the dense
Qwen server smoke. It needs its own exact-profile server receipt before any
server row can promote.

## Work Item: CUDA-SERVER-003

Status: blocked
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

Status: proposed
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007, BITNET-SPEC-0010
Linked ADRs: BITNET-ADR-0004
Campaign item: `CUDA-SERVER-004`
Blocked by: CUDA-SERVER-003 missing receipt fields
Blocks: dense Qwen server status UX

### Goal

Promote dense Qwen2.5 server readiness only for the exact bounded profile after
a refreshed or supplemental receipt carries the artifact checksum, endpoint or
request-profile scope, and generation policy required by BITNET-SPEC-0010.

### Claim Boundary

This cannot claim global dense GGUF server readiness, BitNet QK256 server
readiness, speedup, full CUDA residency, concurrency, or production deployment
readiness.

## Work Item: CUDA-SERVER-005

Status: proposed
Linked proposal: BITNET-PROP-0002
Linked specs: BITNET-SPEC-0007, BITNET-SPEC-0010
Linked ADRs: BITNET-ADR-0004
Campaign item: `CUDA-SERVER-005`
Blocked by: CUDA-SERVER-003
Blocks: official BitNet server status UX

### Goal

Add or promote official BitNet I2_S/QK256 strict server smoke separately from
dense Qwen.

### Claim Boundary

Official BitNet server proof must use `route = bitnet_qk256_cuda`, preserve
QK256 invocation evidence, keep dense regular-LLM proof false, and keep speed
false unless a separate benchmark qualification accepts an exact profile.
