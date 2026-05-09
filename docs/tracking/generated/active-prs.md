<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-023 | #4257 | `codex/cuda-dense-023-attention-softmax-fixture` | Extract a dense GGUF attention-softmax CPU-reference fixture after CUDA-DENSE-022 made attention_scores routable, record probability hashes, causal zero-probability counts, row-sum error, and the missing CUDA softmax kernel gap, while keeping dense GGUF inference, Qwen token/decode/chat, speedup, full-residency, BitNet packed proof, tokenizer, loader, transformer, QK256, server, and CUDA kernel math claims false. |
