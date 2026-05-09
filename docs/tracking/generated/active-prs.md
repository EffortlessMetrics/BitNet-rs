<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-033 | #4318 | `codex/cuda-dense-032-cpu-reference-harness` | Implement the full dense GGUF layer-0 CPU reference harness defined by CUDA-DENSE-032, composing the verified Qwen2.5 0.5B Q8_0 dense GGUF layer phases into a deterministic CPU-only pass with per-phase hashes, final layer output hash, and receipt validation while keeping CUDA execution, dense GGUF inference, Qwen token/decode/chat, speedup, persistent/full residency, BitNet packed proof, tokenizer, loader, transformer, QK256, server, and CUDA kernel math claims false. |
