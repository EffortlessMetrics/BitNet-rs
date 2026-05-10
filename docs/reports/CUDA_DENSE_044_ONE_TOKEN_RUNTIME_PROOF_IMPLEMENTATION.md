# CUDA-DENSE-044 One-Token Runtime Proof Implementation

`CUDA-DENSE-044` implements the governed dense Qwen one-token strict CUDA
runtime proof defined by `CUDA-DENSE-043`.

## What This Proves

The new `dense-gguf-qwen-one-token-strict-cuda` CLI command consumes the
SHA-verified Qwen2.5 0.5B Q8_0 GGUF artifact and the prerequisite all-layer
plan, model-boundary fixture, KV-cache policy, and sampling-policy receipts.
It then runs exactly one deterministic greedy token through both CPU reference
and RTX 5070 Ti CUDA paths and emits a
`dense_gguf_qwen_one_token_strict_cuda_proof` receipt only when the selected
token and top-k rank evidence match.

The command is scoped to the pinned artifact:

```text
model: qwen2.5-0.5b-instruct-q8_0
file: qwen2.5-0.5b-instruct-q8_0.gguf
sha256: ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
```

The CLI `cuda` feature now propagates CUDA support to the common/model loader
path so the proof command can construct a real Candle CUDA device instead of a
CPU device while leaving loader, transformer, tokenizer, and kernel math
behavior unchanged.

## Committed Hardware Receipts

The runtime proof receipt was emitted on the Windows 9950X3D + RTX 5070 Ti
machine:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json
```

The all-layer execution-plan prerequisite receipt was also refreshed from the
same verified artifact:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-all-layer-plan-qwen25-q8.json
```

Observed one-token receipt summary:

```text
artifact_kind: dense_gguf_qwen_one_token_strict_cuda_proof
selected_backend: nvidia-rtx-5070-ti-cuda
execution_plan.selected_route: dense_regular_llm_cuda
cuda_dense_regular_llm_ops: 338
unsupported_ops: 0
fallback_used: false
cpu_selected_token_id: 576
cuda_selected_token_id: 576
top_k_match: true
speedup_claim: false
qwen_short_decode_cuda_claimed: false
qwen_chat_cuda_claimed: false
bitnet_packed_i2s_qk256_proof: false
```

## Validation

Run locally from a Visual Studio developer environment with CUDA toolkit
`v12.9` on `PATH`:

```text
cargo check --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli

cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-qwen-one-token-strict-cuda --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf --prompt "What is 2+2?" --top-k 10 --device-index 0 --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-qwen-one-token-strict-cuda-qwen25-q8.json
```

The proof command validates the generated receipt with
`validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json` before
writing it.

## Claim Boundary

May claim:

```text
Qwen one deterministic greedy token executed through dense_regular_llm_cuda on RTX 5070 Ti CUDA.
fallback_used=false.
CPU and CUDA selected token IDs matched.
CPU and CUDA top-k token-rank evidence matched.
The receipt links all-layer plan, model-boundary fixture, KV-cache policy, and sampling-policy prerequisite receipts.
```

Must not claim:

```text
Qwen short decode or chat works on CUDA.
Dense GGUF inference is generally complete.
CUDA speedup is benchmark-qualified.
Persistent-session or full dense CUDA residency exists.
Server readiness exists.
Dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference.
Tokenizer, prompt-template, loader, transformer runtime, server, QK256, BitNet CUDA, or CUDA kernel math changed.
```
