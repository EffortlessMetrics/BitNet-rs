<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU258V-021 | #4315 | `codex/lunar-lake/CPU258V-021-reference-instrumentation` | Instrument or script the external BitNet reference runner boundary so the fixed 258V prompts expose prompt token IDs, first generated token IDs, decoded first tokens, and first-token logits/top-k when available; if the reference cannot expose logits or token IDs, record the blocker precisely without inferring parity. |
| nvidia-5070ti | CUDA-DENSE-032 | #4318 | `codex/cuda-dense-032-cpu-reference-harness` | Implement the full dense GGUF layer-0 CPU reference harness after CUDA-DENSE-031 made every governed one-layer op CUDA-routable, composing the verified Qwen2.5 0.5B Q8_0 dense GGUF layer phases into a deterministic CPU-only pass with per-phase hashes, final layer output hash, and receipt validation while keeping CUDA execution, dense GGUF inference, Qwen token/decode/chat, speedup, persistent/full residency, BitNet packed proof, tokenizer, loader, transformer, QK256, server, and CUDA kernel math claims false. |
