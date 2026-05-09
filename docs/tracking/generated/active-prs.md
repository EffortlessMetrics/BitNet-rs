<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU258V-019 | #4248 | `codex/lunar-lake/CPU258V-019-external-first-token-reference` | Capture external bitnet.cpp or HF reference prompt policy, prompt token IDs, first generated token ID when available, decoded first token, generated text, and explicit missing-logits status for the fixed 258V prompts without claiming logits parity unless the reference exposes logits. |
| nvidia-5070ti | CUDA-DENSE-027 | #4290 | `codex/cuda-dense-027-attention-v-mix-parity` | Run the CUDA-DENSE-026 attention V-mix fixture through a strict RTX 5070 Ti CUDA F32 V-mix kernel, prove parity against the CPU-reference context vectors, record kernel/transfer/residency receipt fields, and keep dense GGUF inference, Qwen token/decode/chat, speedup, full-residency, BitNet packed proof, tokenizer, loader, transformer, QK256, server, and planner route-promotion claims false. |
