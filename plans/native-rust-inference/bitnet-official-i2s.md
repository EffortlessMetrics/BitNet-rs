# Official BitNet I2_S/QK256

Official BitNet 2B I2_S/QK256 is product CLI-ready on RTX 5070 Ti CUDA and has
BitNet QK256 proof. Speedup, full residency, and broad server readiness remain
false.

## Work item: CUDA-BITNET-PERF-005

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0003-native-rust-inference-product.md`
Linked specs: `docs/specs/BITNET-SPEC-I2S-CUDA.md`, `docs/specs/BITNET-SPEC-I2S-STATUS-SURFACE.md`, `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`
Linked ADRs: `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
Campaign: `docs/tracking/campaigns/nvidia-5070ti/active.toml`
Blocks: CUDA-BITNET-PERF-006, CUDA-BITNET-OPS-001
Blocked by: none

### Goal

Run repeated current-source profiles for the official Microsoft BitNet 2B
I2_S/QK256 artifact on the canonical 9950X3D CPU reference and RTX 5070 Ti
CUDA product bench.

### Production delta

Receipts and a review report cover the official artifact across governed
profiles with the same tokenizer/prompt authority, CPU AVX-512 comparator,
strict RTX 5070 Ti CUDA comparator, selected route, QK256 kernel counters,
transfer/timing fields, and preserved false speed/full-residency/server claims.

### Non-goals

- no speedup promotion;
- no `benchmark_qualified=true` promotion;
- no full-residency promotion;
- no broad server-readiness promotion;
- no dense SLM proof;
- no tokenizer, loader, transformer, QK256 math, kernel, or server runtime
  change.

### Acceptance

The PR must add current-source receipt evidence, or a report that points to the
exact receipt paths, for these profiles:

```text
one_token
short_decode_8
short_decode_32
prefill_128_decode_16
prefill_512_decode_32
warm_session_3_turns
warm_session_10_turns
decode_128_from_warm_context
```

Each profile must record:

```text
model_artifact = official Microsoft I2_S/QK256 GGUF
model_coverage_row = bitnet_official_2b_i2s_qk256
tokenizer_authority = external Llama BPE
prompt_template = bitnetcpp-answer
requested_backend
selected_backend = nvidia-rtx-5070-ti-cuda for CUDA receipts
runtime_api = cuda for CUDA receipts
selected_route = bitnet_qk256_cuda for CUDA receipts
fallback_used = false
unsupported_strict_ops = 0
bitnet_linear_cpu_fallback_ops = 0
selected_kernel_id or qk256_gemv_cuda summary
qk256_gemv_cuda invocation count
model_load_ms
tokenizer_load_ms
prompt_render_ms
tokenize_ms
cuda_context_init_ms
weight_upload_ms
prefill_ms
first_token_ms
decode_total_ms
steady_tok_per_s
kernel_time_ms
launch_count
H2D_bytes and H2D_ms, or explicit timing-source limitation
D2H_bytes and D2H_ms, or explicit timing-source limitation
VRAM_high_water
power_temperature_context
speedup_claim = false
full_residency_claim = false
server_ready = false
dense_regular_llm_cuda_proof = false
```

The report must identify missing fields as blockers rather than converting
unknowns to false zeros. Model coverage rows and generated dashboards must not
be hand-edited in this PR.

### Proof commands

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --model microsoft-bitnet-b1.58-2B-4T-i2s --profile one_token
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --model microsoft-bitnet-b1.58-2B-4T-i2s --profile short_decode_8
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --model microsoft-bitnet-b1.58-2B-4T-i2s --profile short_decode_32
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --model microsoft-bitnet-b1.58-2B-4T-i2s --profile prefill_128_decode_16
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --model microsoft-bitnet-b1.58-2B-4T-i2s --profile prefill_512_decode_32
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --model microsoft-bitnet-b1.58-2B-4T-i2s --profile warm_session_3_turns
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --model microsoft-bitnet-b1.58-2B-4T-i2s --profile warm_session_10_turns
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda --model microsoft-bitnet-b1.58-2B-4T-i2s --profile decode_128_from_warm_context
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain <each-new-receipt> --format json
cargo run --locked -p xtask --no-default-features -- check-model-coverage
git diff --check
```

### Rollback

Revert CUDA-BITNET-PERF-005 receipts and report. Keep the existing
`product_cli_ready` BitNet row, strict CUDA proof state, and false speed,
full-residency, and broad server-readiness claims unchanged.

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
