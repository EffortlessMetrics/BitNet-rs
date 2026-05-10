# CUDA-DENSE-046 Warm-Session Runtime Proof Implementation

## Scope

`CUDA-DENSE-046` extends the governed dense Qwen strict CUDA runtime from the
short-decode proof to a bounded warm-session proof. The implementation adds a
single command:

```text
bitnet dense-gguf-qwen-warm-session-strict-cuda
```

The command is scoped to the SHA-verified `qwen2.5-0.5b-instruct-q8_0.gguf`
artifact on the Windows 9950X3D + RTX 5070 Ti lane.

## What It Proves

The committed hardware receipt proves:

```text
artifact_kind: dense_gguf_qwen_warm_session_strict_cuda_proof
selected_route: dense_regular_llm_cuda
selected_backend: nvidia-rtx-5070-ti-cuda
fallback_used: false
turns: 3
generated_tokens_total: 24
generated_token_ids_match: true
top_k_all_match: true
qwen_warm_session_cuda_claimed: true
qwen_chat_cuda_claimed: false
speedup_claim: false
full_cuda_residency_claimed: false
bitnet_packed_i2s_qk256_proof: false
```

The receipt records prerequisite receipt hashes for the all-layer plan,
model-boundary fixtures, KV-cache policy, sampling policy, one-token proof, and
short-decode proof. It also records tokenizer/prompt authority, per-turn prompt
token hashes, generated-token evidence, top-k/logit evidence, kernel summaries,
timing, transfer bytes, and scoped warm-session reuse evidence.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-warm-session-strict-cuda-qwen25-q8.json
```

Per-turn generated CUDA tokens:

```text
turn 0: 576,4226,374,220,19,13,3555,374
turn 1: 576,1894,315,279,12884,374,6303,13
turn 2: 7684,6556,0,2585,646,358,7789,498
```

Decoded text:

```text
turn 0:  The answer is 4. What is
turn 1:  The color of the sky is blue.
turn 2: Good morning! How can I assist you
```

## Non-Claims

This PR does not claim:

```text
Qwen ask/chat UX
general dense GGUF inference beyond the bounded warm-session proof
speedup
full CUDA residency
persistent residency beyond the scoped warm-session receipt
server readiness
BitNet packed I2_S/QK256 proof
QK256 behavior changes
tokenizer, loader, transformer, or CUDA kernel math changes
```

## Validation

The implementation is expected to validate with:

```text
cargo fmt -p bitnet-cli -p bitnet-receipts-core -p bitnet-receipts -- --check
cargo check --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features qwen_warm_session -- --nocapture
cargo test --locked -p bitnet-receipts --no-default-features
cargo test --locked -p bitnet-cli --bin bitnet --no-default-features --features cpu,cuda,full-cli -- --nocapture
cargo run --release --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --release --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```
