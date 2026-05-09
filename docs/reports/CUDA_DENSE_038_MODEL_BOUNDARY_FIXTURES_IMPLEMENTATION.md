# CUDA-DENSE-038 Model-Boundary Fixtures Implementation

`CUDA-DENSE-038` adds the governed dense GGUF model-boundary fixture receipt
path after the all-layer transformer plan is strict-CUDA route complete.

## What This Proves

The new `dense_gguf_model_boundary_fixtures` receipt records:

```text
token embedding lookup fixture
final model norm fixture
LM head / output projection fixture
logits length and hash
deterministic logits top-k diagnostics
selected route = dense_regular_llm_cuda
fallback_used = false
```

The implementation keeps the fixture route scoped to model-boundary evidence.
It does not execute a CUDA kernel for these boundary fixtures and does not make
an inference claim.

## Claim Boundary

May claim:

```text
dense GGUF model-boundary fixtures are recorded
token embedding, final norm, and LM head/logits fixture receipts validate
the selected dense_regular_llm_cuda route boundary is visible
KV cache and sampling remain explicit blockers for one-token proof
```

Must not claim:

```text
Qwen one-token CUDA proof
Qwen short decode or chat
general dense GGUF CUDA inference
speedup
persistent-session or full CUDA residency
KV cache policy
sampling integration
BitNet packed I2_S / QK256 proof
tokenizer, loader, transformer runtime, server, QK256, BitNet CUDA, or CUDA kernel math changes
```

## Validation

Run locally:

```text
cargo fmt --package bitnet-cli --package bitnet-receipts-core --package bitnet-receipts -- --check
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features model_boundary -- --nocapture
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli dense_gguf_model_boundary -- --nocapture
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli dense_gguf_all_layer -- --nocapture
cargo check --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
```

Hardware receipt:

```text
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-model-boundary-fixtures --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf --device-index 0 --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-model-boundary-fixtures-qwen25-q8.json
```

The source artifact was verified first:

```text
id = qwen2.5-0.5b-instruct-q8_0
sha256 = ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
bytes = 675710816
passed = true
```

The committed hardware receipt records RTX 5070 Ti CUDA availability and the
selected `dense_regular_llm_cuda` route boundary. It still makes no CUDA kernel
execution, dense inference, one-token, decode, chat, speedup, KV cache,
sampling, persistent-residency, full-residency, or BitNet packed proof claim.

## Completion Audit

Objective slice:

```text
Implement governed dense GGUF model-boundary fixture receipts covering token
embedding lookup, final model norm, LM head/output projection, logits
shape/hash/top-k diagnostics, and the selected dense_regular_llm_cuda route
boundary.
```

Checklist:

| Requirement | Evidence | Status |
| --- | --- | --- |
| CLI command exists | `DenseGgufModelBoundaryFixturesCommand` and `dense-gguf-model-boundary-fixtures` dispatch | covered |
| Token embedding lookup fixture | `dense_gguf_model_boundary_fixtures_from_reader` materializes `token_embd.weight` and deterministic token ids | covered by unit test |
| Final model norm fixture | final norm descriptor aliases include `output_norm.weight`; receipt records input/output hashes and epsilon source | covered by unit test |
| LM head/output projection fixture | output role extraction feeds final norm output through `dense_linear_sequence_cpu` | covered by unit test |
| Logits shape/hash/top-k diagnostics | receipt records `logits_len`, `logits_sha256`, `top_k`, and ranked entries | covered by unit and receipt tests |
| Selected dense CUDA route boundary | receipt `execution_plan.selected_route = dense_regular_llm_cuda`, `cuda_dense_regular_llm_ops = 3` | covered by validator |
| Fallback remains false | receipt and validator require `fallback_used = false` | covered by validator |
| No CUDA kernel execution claim | fixture receipt requires `cuda_kernel_execution_claimed = false` and `kernel_invocations = 0` | covered by validator |
| No KV or sampling claim | validator rejects `kv_cache_policy_claimed` and `sampling_integration_claimed` | covered by rejection tests |
| No Qwen one-token/decode/chat claim | validator rejects `qwen_one_token_cuda_claimed`; receipt keeps decode/chat false | covered by rejection tests |
| No speedup/full residency/BitNet proof claim | validator requires all false and rejects dense receipt as BitNet packed proof | covered by rejection tests |
| Real verified Qwen artifact receipt | `dense-gguf-model-boundary-fixtures-qwen25-q8.json` emitted from the SHA-verified Qwen2.5 Q8_0 cache artifact | covered |

The implementation slice is locally test-covered and artifact-backed for the
model-boundary fixture receipt. The next proof gate remains KV cache and
sampling policy before any Qwen one-token CUDA claim.
