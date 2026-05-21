# Official BitNet I2_S/QK256

Official BitNet 2B I2_S/QK256 is product CLI-ready on RTX 5070 Ti CUDA and has
BitNet QK256 proof. Speedup, full residency, and broad server readiness remain
false.

## Work item: CUDA-BITNET-PERF-005

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-BITNET-PERF-006, CUDA-BITNET-OPS-001
Blocked by: native inference plan

### Goal

Run repeated current-source profiles for the official Microsoft I2_S artifact.

### Production delta

Receipts cover one-token, short decode, prefill/decode, warm-session, and warm
decode profiles for CPU AVX-512 and RTX 5070 Ti CUDA comparators.

### Non-goals

No global speedup, dense proof, broad server claim, or full-residency claim.

### Acceptance

Receipts include same artifact/tokenizer/prompt authority, fallback false,
TTFT, prefill, decode, kernel time, launch count, H2D/D2H timing, VRAM
high-water, and `speedup_claim=false`.

### Proof commands

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --model <official-i2s> --profile short_decode_32
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- receipts explain --latest --format json
```

### Rollback

Revert benchmark review artifacts and keep existing product CLI state.

## Work item: CUDA-BITNET-PERF-006

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: exact-profile status updates
Blocked by: CUDA-BITNET-PERF-005

### Goal

Accept or reject official BitNet speed by exact profile.

### Production delta

Only profiles accepted by governed review may set speed-related claim state.

### Non-goals

No global speedup, dense proof, broad server claim, or full-residency claim.

### Acceptance

Review records accepted and rejected profiles with reasons.

### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```

### Rollback

Set speed claim booleans back to false for rejected or unsupported profiles.

## Work item: CUDA-BITNET-OPS-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-BITNET-OPS-002
Blocked by: CUDA-BITNET-PERF-005

### Goal

Audit TTFT and residency bottlenecks for official BitNet.

### Production delta

Report where KV, norm, RoPE, attention, LM head, H2D/D2H transfer, first token,
and warm decode time are spent.

### Non-goals

No optimization in the audit PR.

### Acceptance

Audit ranks the next op or transfer that should move.

### Proof commands

```bash
git diff --check
```

### Rollback

Revert the report.

## Work item: CUDA-BITNET-OPS-002

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: requalification review
Blocked by: CUDA-BITNET-OPS-001

### Goal

Add persistent session optimization.

### Production delta

Model, CUDA context, weights, and workspace are reused where the route supports
it.

### Non-goals

No speedup claim until review.

### Acceptance

Receipt shows `model_loaded_once=true`, `cuda_context_once=true`,
`weights_uploaded_once=true`, `workspace_reused=true`,
`per_request_model_load=false`, and `fallback_used=false`.

### Proof commands

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- chat --device cuda --model <official-i2s>
```

### Rollback

Disable persistent handles and return to prior request lifecycle.
