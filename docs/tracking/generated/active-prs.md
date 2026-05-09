<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-025 | #4269 | `codex/cuda-dense-025-attention-softmax-route` | Update the dense GGUF one-layer planner and gap receipt to mark verified attention_softmax as dense_regular_llm_cuda routable after CUDA-DENSE-024, reducing unsupported strict CUDA gaps to attention_v_mix and mlp_activation while keeping dense GGUF inference, Qwen token/decode/chat, speedup, full-residency, BitNet packed proof, tokenizer, loader, transformer, QK256, server, and CUDA kernel math claims false. |
