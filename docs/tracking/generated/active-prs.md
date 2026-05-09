<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-031 | #4314 | `codex/cuda-dense-031-mlp-activation-route` | Promote verified dense GGUF mlp_activation to the one-layer planner route after CUDA-DENSE-030, refresh the one-layer execution-plan receipt to 14 dense_regular_llm_cuda ops, unsupported_ops=0, strict_cuda_ready=true, and keep dense GGUF inference, Qwen token/decode/chat, speedup, full-residency, BitNet packed proof, tokenizer, loader, transformer, QK256, server, and CUDA kernel math claims false. |
