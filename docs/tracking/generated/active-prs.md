<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-018 | #4228 | `codex/cuda-dense-018-rope-parity` | Add a strict RTX 5070 Ti CUDA RoPE fixture parity proof for the verified Qwen2.5 0.5B Q8_0 dense GGUF metadata, using metadata-derived Q/K head counts and RoPE parameters, recording dense_rope_f32_cuda kernel launches, transfer accounting, CPU reference comparison, and claim boundaries while keeping dense GGUF inference, Qwen one-token/decode/chat, speedup, full-residency, BitNet packed proof, tokenizer, loader, transformer, QK256, and server claims false. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
