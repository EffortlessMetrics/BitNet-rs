<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU258V-019 | #4248 | `codex/lunar-lake/CPU258V-019-external-first-token-reference` | Capture external bitnet.cpp or HF reference prompt policy, prompt token IDs, first generated token ID when available, decoded first token, generated text, and explicit missing-logits status for the fixed 258V prompts without claiming logits parity unless the reference exposes logits. |
| nvidia-5070ti | CUDA-DENSE-026 | #4281 | `codex/cuda-dense-026-attention-v-mix-fixture` | Extract a dense GGUF attention V-mix CPU-reference fixture after CUDA-DENSE-025 made attention_softmax CUDA-routable, record context-vector hashes, dependency authority for attention_softmax and attention_v, and the missing CUDA V-mix kernel gap, while keeping dense GGUF inference, Qwen token/decode/chat, speedup, full-residency, BitNet packed proof, tokenizer, loader, transformer, QK256, server, and CUDA kernel math claims false. |
