# CUDA-DENSE-045 Short-Decode Runtime Proof Implementation

`CUDA-DENSE-045` extends the governed dense Qwen runtime proof from one
deterministic greedy token to a bounded short decode.

## What This Proves

The new `dense-gguf-qwen-short-decode-strict-cuda` CLI command consumes the
SHA-verified Qwen2.5 0.5B Q8_0 GGUF artifact and the prerequisite all-layer
plan, model-boundary fixture, KV-cache policy, sampling-policy, and one-token
proof receipts. It then runs 5-16 deterministic greedy tokens through both the
CPU reference path and the RTX 5070 Ti CUDA path and emits a
`dense_gguf_qwen_short_decode_strict_cuda_proof` receipt only when generated
token IDs match.

The command is scoped to the pinned artifact:

```text
model: qwen2.5-0.5b-instruct-q8_0
file: qwen2.5-0.5b-instruct-q8_0.gguf
sha256: ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
```

## Committed Hardware Receipt

The short-decode proof receipt was emitted on the Windows 9950X3D + RTX 5070 Ti
machine:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json
```

Observed receipt summary:

```text
artifact_kind: dense_gguf_qwen_short_decode_strict_cuda_proof
selected_backend: nvidia-rtx-5070-ti-cuda
execution_plan.selected_route: dense_regular_llm_cuda
fallback_used: false
generated_tokens: 576,4226,374,220,19,13,3555,374
decoded_text: " The answer is 4. What is"
top_k_all_match: true
speedup_claim: false
qwen_chat_cuda_claimed: false
full_cuda_residency_claimed: false
bitnet_packed_i2s_qk256_proof: false
```

## Validation

Run locally from a Visual Studio developer environment with CUDA toolkit
`v12.9` on `PATH`:

```text
cargo check --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli

cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features qwen_short_decode -- --nocapture

cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-qwen-short-decode-strict-cuda --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf --prompt "What is 2+2?" --max-new-tokens 8 --top-k 10 --device-index 0 --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-qwen-short-decode-strict-cuda-qwen25-q8.json
```

The proof command validates the generated receipt with
`validate_dense_gguf_qwen_short_decode_strict_cuda_proof_receipt_json` before
writing it.

## Claim Boundary

May claim:

```text
Qwen bounded deterministic greedy short decode executed through dense_regular_llm_cuda on RTX 5070 Ti CUDA.
fallback_used=false.
CPU and CUDA generated token IDs matched for the bounded proof.
Top-k/logit evidence was recorded per decode step.
The receipt links all-layer plan, model-boundary fixture, KV-cache policy, sampling-policy, and one-token proof prerequisite receipts.
```

Must not claim:

```text
Qwen chat works on CUDA.
Dense GGUF inference is generally complete beyond the bounded short-decode gate.
CUDA speedup is benchmark-qualified.
Persistent-session or full dense CUDA residency exists.
Server readiness exists.
Dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference.
Tokenizer, prompt-template, loader, transformer runtime, server, QK256, BitNet CUDA, or CUDA kernel math changed.
```
