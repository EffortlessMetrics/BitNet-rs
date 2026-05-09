<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-016 | #4220 | `codex/cuda-dense-016-rmsnorm-parity` | Add strict RTX 5070 Ti CUDA parity for dense GGUF RMSNorm fixtures by routing the verified Qwen2.5 0.5B Q8_0 attention_norm and ffn_norm CPU reference fixtures through a dense_rmsnorm_f32_cuda kernel, validating a dense_gguf_norm_cuda_parity receipt, and keeping dense GGUF inference, Qwen one-token/decode/chat, speedup, full-residency, BitNet packed proof, tokenizer, loader, transformer, QK256, and server claims false. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
