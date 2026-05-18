# BITNET-SPEC-0010: Server Readiness Proof Boundary

Status: proposed
Linked proposal:
[BITNET-PROP-0002](../proposals/BITNET-PROP-0002-9950x3d-5070ti-cuda-productization.md)
Linked specs:
[BITNET-SPEC-0001](BITNET-SPEC-0001-source-of-truth-and-claim-boundaries.md),
[BITNET-SPEC-0007](BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md)
Linked ADRs:
[BITNET-ADR-0004](../adr/BITNET-ADR-0004-9950x3d-5070ti-cuda-product-bench.md)
Applies to: RTX 5070 Ti CUDA server smoke, exact-profile server readiness
promotion, model coverage `server_ready` claims, receipt explanation, CUDA
status docs

## Purpose

Server execution is a separate product surface from CLI `ask`, `chat`,
warm-session, and benchmark receipts. A fallback-free CLI receipt can prove a
model route, backend, quality gate, or benchmark profile without proving that
the server path is ready for users.

This spec defines the boundary for promoting `server_ready=true` in
`ci/model-artifacts/model-coverage-matrix.toml`. It keeps bounded server smoke
useful while preventing one smoke receipt from becoming a broad production
serving claim.

## Source-Of-Truth Authorities

Server readiness promotion must use these authorities instead of copying their
full tables into this spec:

- [CUDA product contract](BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md)
- [CUDA capability matrix](../status/CUDA_CAPABILITY_MATRIX.md)
- [CUDA receipt triage guide](../tutorials/cuda-receipt-triage.md)
- [NVIDIA 5070 Ti campaign](../tracking/campaigns/nvidia-5070ti/CAMPAIGN.md)
- `ci/model-artifacts/model-coverage-matrix.toml`
- `ci/hardware/windows-9950x3d-rtx5070ti/**`

The model coverage matrix owns the final `server_ready` boolean. Server
receipts prove what happened. Status docs and tutorials summarize the current
state, but they do not promote server readiness by themselves.

## Requirements

### 1. Server Readiness Is Exact-Profile Only

`server_ready=true` may apply only to the exact model, artifact, route,
backend, endpoint shape, prompt policy, generation policy, and profile proven
by receipts.

A server-ready row for one model does not imply:

- server readiness for another model in the same family;
- server readiness for another quantization of the same model;
- server readiness for another backend;
- server readiness for another endpoint or serving mode;
- production serving capacity, concurrency, uptime, or deployment hardening.

### 2. Bounded Server Smoke Is Not Promotion

A bounded server smoke receipt can prove that one server path answered one
scoped request with the recorded route and backend. It does not promote
`server_ready=true` unless a separate promotion PR checks this spec, updates the
model coverage row, and explains the exact profile being promoted.

The dense Qwen2.5 strict server-smoke receipt is useful evidence. Until a
promotion PR explicitly accepts it, the correct product status remains
`server_ready=false`.

### 3. Server Receipt Invariants

Every receipt used to promote RTX 5070 Ti CUDA server readiness must preserve:

```text
model_id
artifact identity and checksum
model coverage row
server endpoint or internal server path
request schema or endpoint profile
requested_backend = nvidia-rtx-5070-ti-cuda
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
route = bitnet_qk256_cuda | dense_regular_llm_cuda
fallback_used = false
tokenizer authority present
prompt template authority present
generation policy present
quality gate result present
response non-empty and valid UTF-8 for answer claims
receipt id or durable receipt path
speedup_claim = false unless separately benchmark-qualified
full_residency_claim = false unless separately proven
server_ready_claimed must not be true outside the exact promoted profile; the
model coverage row owns the final server_ready promotion decision
```

BitNet server receipts must also preserve BitNet route evidence:

```text
bitnet_packed_i2s_qk256_proof = true
dense_regular_llm_cuda_proof = false
qk256 kernel invocation evidence present
BitNet linear CPU fallback count = 0
```

Dense SLM or small dense LLM server receipts must preserve dense route evidence:

```text
dense_regular_llm_cuda_proof = true only for the exact artifact
bitnet_packed_i2s_qk256_proof = false
```

Server receipts must not mark the dense route/proof or populate a dense
model-coverage row unless the exact dense profile artifact identity matches and
the active model was loaded through CUDA. Otherwise, the shared-engine receipt
stays generic and no dense readiness or proof claim is emitted.

### 4. Fallback Must Fail Closed

Strict server readiness cannot allow hidden fallback. A promotion is blocked if
the receipt records:

- selected backend different from `nvidia-rtx-5070-ti-cuda`;
- generic `cuda` without strict selected backend identity;
- CPU fallback;
- unsupported route fallback;
- missing tokenizer or prompt authority;
- missing model artifact identity;
- missing quality gate result for answer claims.

### 5. CLI Proof Does Not Substitute For Server Proof

CLI `ask`, `chat`, warm-session, and benchmark receipts may be prerequisites
for server promotion. They are not server proof by themselves.

A server promotion PR must identify whether the server path uses the same engine
as the CLI path or a distinct serving adapter. If there is an engine or adapter
delta, the receipt or promotion notes must say which part is shared and which
part is server-specific.

### 6. Status And Receipt Explanation Must Agree

When a row is promoted to `server_ready=true`, the user-facing surfaces must
agree:

- `ci/model-artifacts/model-coverage-matrix.toml`
- `docs/model-artifacts/MODEL_COVERAGE_MATRIX.md` when updated
- `docs/status/CUDA_CAPABILITY_MATRIX.md`
- `bitnet model status --device nvidia-rtx-5070-ti-cuda`
- `bitnet receipts explain <server-receipt>` when the receipt is available

If these surfaces disagree, keep the narrower claim until the mismatch is
repaired.

## Promotion Rule

A promotion PR may set `server_ready=true` only when it includes:

1. exact server receipt path or receipt id;
2. exact model coverage row;
3. exact model artifact identity;
4. exact route and selected backend;
5. fallback-free server request proof;
6. quality gate result;
7. statement of endpoint/profile scope;
8. explicit non-claims for speed, full residency, broad server readiness, and
   cross-family proof;
9. status or model coverage updates that match the receipt.

If any required field is unavailable, keep `server_ready=false` and record the
next missing proof instead.

## Claim Boundary

| Evidence | May claim | Must not claim |
| --- | --- | --- |
| CLI ask/chat receipt | scoped CLI answer path | server readiness |
| Warm-session receipt | scoped reusable CLI/session behavior | server endpoint readiness |
| Bounded server smoke | one scoped server path answered with recorded backend and route | broad production serving readiness |
| Exact-profile server promotion | `server_ready=true` for that model/profile only | global dense GGUF, BitNet, speed, concurrency, or deployment readiness |
| Dense server receipt | dense regular-LLM server route for the exact artifact | BitNet QK256 server proof |
| BitNet server receipt | BitNet QK256 server route for the exact artifact | dense SLM server proof |

## Proof Commands

This spec is documentation-only. Its validation is:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --locked -p xtask --no-default-features -- campaign generate --check
```

Runtime PRs that promote server readiness must add the exact server command or
internal test path they use, plus receipt validation. For example:

```bash
cargo run --locked -p bitnet-server --no-default-features --features cpu,cuda -- --device nvidia-rtx-5070-ti-cuda --model <verified-model>
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features server_shared_engine_chat_completion
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain <server-receipt>
```

## Non-Goals

- Do not implement server runtime behavior in this spec.
- Do not modify receipts in this spec.
- Do not promote any model coverage row in this spec.
- Do not claim dense Qwen server readiness from server smoke alone.
- Do not claim official BitNet server readiness from dense Qwen server proof.
- Do not claim speedup, full residency, concurrency, deployment readiness, or
  production service reliability.
- Do not make GPU, model, Windows, macOS, Docker, or server lanes default PR CI.

## Related Policy Or Manifest Sources

- `ci/model-artifacts/model-coverage-matrix.toml`
- `docs/status/CUDA_CAPABILITY_MATRIX.md`
- `docs/tracking/campaigns/nvidia-5070ti/active.toml`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/server-strict-dense-qwen25-q8-smoke.json`
- `policy/ci-lanes.toml`
- `policy/ci-budget.toml`
- `policy/ci-risk-packs.toml`
