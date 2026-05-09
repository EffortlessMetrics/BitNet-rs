<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-035 | #4328 | `codex/cuda-dense-035-one-layer-cuda-parity` | Implement the governed integrated dense GGUF one-layer CUDA parity harness defined by CUDA-DENSE-034, composing the verified layer-0 CPU reference phases through the CUDA-routable dense_regular_llm_cuda plan, comparing per-phase and final output parity, recording per-op kernel stats and aggregate H2D/D2H transfer accounting, and keeping dense GGUF inference, Qwen token/decode/chat, speedup, persistent/full residency, BitNet packed proof, tokenizer, loader, transformer, QK256, server, and CUDA kernel math claims false. |
