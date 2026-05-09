<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-slm-model-breadth | M4-MODEL-003 | #4355 | `codex/apple-m4-slm-model-breadth-M4-MODEL-003-rust-m4-quality` | Run the candidate through Rust M4 apple-m4-cpu-neon quality gates with valid UTF-8, non-empty output, non-degenerate output, backend/fallback receipts, generated token IDs, timing, and deterministic behavior where required. |
| nvidia-5070ti | CUDA-DENSE-039 | #4357 | `codex/cuda-dense-039-kv-policy` | Implement governed dense GGUF KV-cache policy receipts after CUDA-DENSE-038, recording prefill KV write policy, decode KV read/write policy, estimated bytes per token/layer/all layers, planned strict CUDA residency, remaining sampling gap, and claim-boundary rejection of runtime KV allocation, dense GGUF inference, Qwen one-token/short decode/chat, speedup, persistent/full residency, server readiness, BitNet packed proof, tokenizer behavior, loader behavior, transformer runtime behavior, QK256, BitNet CUDA, and CUDA kernel math claims. |
