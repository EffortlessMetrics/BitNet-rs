<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-034 | #4324 | `codex/cuda-dense-034-integrated-parity-tracker` | Define the governed integrated dense GGUF one-layer CUDA parity contract after CUDA-DENSE-033, requiring the future implementation to run the full layer-0 CUDA-routable plan against the same Qwen2.5 0.5B Q8_0 CPU reference harness, compare per-phase and final output parity, record per-op kernel stats and aggregate H2D/D2H transfer accounting, and keep dense GGUF inference, Qwen token/decode/chat, speedup, persistent/full residency, BitNet packed proof, tokenizer, loader, transformer, QK256, server, and CUDA kernel math claims false. |
