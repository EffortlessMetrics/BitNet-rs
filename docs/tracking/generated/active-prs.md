<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-015 | #4218 | `codex/cuda-dense-015-norm-fixture-audit` | Extract dense GGUF RMSNorm fixtures for attention_norm and ffn_norm from the verified Qwen2.5 0.5B Q8_0 artifact, compute deterministic CPU RMSNorm reference outputs, validate a dense_gguf_norm_fixture_extraction receipt, and record the current CUDA norm gap as missing_cuda_kernel while keeping dense GGUF inference, Qwen token/decode/chat, speedup, full-residency, BitNet packed proof, tokenizer, loader, transformer, kernel, and server claims false. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
