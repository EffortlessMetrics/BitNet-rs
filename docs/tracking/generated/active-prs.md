<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-017 | #4222 | `codex/cuda-dense-017-one-layer-rmsnorm-route` | Update the dense GGUF one-layer planner and gap receipt to mark verified attention_norm and ffn_norm RMSNorm ops as dense_regular_llm_cuda routable after CUDA-DENSE-016, reducing unsupported strict CUDA gaps to RoPE, attention score/softmax/V mix, and MLP activation while keeping dense GGUF inference, Qwen one-token/decode/chat, speedup, full-residency, BitNet packed proof, tokenizer, loader, transformer, QK256, and server claims false. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
