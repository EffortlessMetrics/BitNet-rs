<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-slm-performance | M4-SLM-PERF-003 | #4050 | `codex/apple-m4-slm-performance/M4-SLM-PERF-003-resident-session` | Make resident model/tokenizer reuse the normal multi-prompt path, with session-owned buffers, safe runtime-state reuse, per-prompt receipts, and an aggregate receipt separating model_load, tokenize, prefill, decode, sampling, and total timing. |
| intel-258v-platform | CPU258V-015 | #4046 | `codex/intel-258v-platform/CPU258V-015-post-mechanics-corpus` | Record post-mechanics release-built 258V scalar and AVX2 answer-corpus receipts for the full committed strict-bitnet-answer-corpus-v1 prompt set, showing all five fixed cases pass and scalar-vs-AVX2 full-corpus parity holds after the RMSNorm/ReLU2 and tied-output-head corrections without general chat, speed, Arc, or NPU claims. |
| nvidia-5070ti | CUDA-ANSWER-012 | #4048 | `codex/cuda-answer/CUDA-ANSWER-012-logit-divergence` | Align the RTX 5070 Ti CUDA QK256 inline-scale path with BitNet.cpp I2_S x I8_S activation semantics so same-box CPU AVX-512 and RTX 5070 Ti CUDA generated token IDs and decoded text match for the five committed deterministic corpus cases while preserving remaining top-k-only divergence evidence without speed or broad-chat claims. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
