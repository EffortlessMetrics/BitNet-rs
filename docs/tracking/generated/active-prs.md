<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU258V-025 | #4356 | `codex/lunar-lake/CPU258V-025-transformer-layer-parity` | Add a 258V CPU transformer-layer parity ladder that records or classifies the first internal divergence across embedding, normalization, Q/K/V projection, RoPE, attention, FFN/ReLU2, residual, final norm, and lm_head boundaries after prompt/token, QK256 semantics, output-head, and logits-index checks are recorded. |
| nvidia-5070ti | CUDA-DENSE-039 | #4357 | `codex/cuda-dense-039-kv-policy` | Implement governed dense GGUF KV-cache policy receipts after CUDA-DENSE-038, recording prefill KV write policy, decode KV read/write policy, estimated bytes per token/layer/all layers, planned strict CUDA residency, remaining sampling gap, and claim-boundary rejection of runtime KV allocation, dense GGUF inference, Qwen one-token/short decode/chat, speedup, persistent/full residency, server readiness, BitNet packed proof, tokenizer behavior, loader behavior, transformer runtime behavior, QK256, BitNet CUDA, and CUDA kernel math claims. |
