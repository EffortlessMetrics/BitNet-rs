<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-013 | #4214 | `codex/cuda-dense-013-one-layer-plan` | Add a dense GGUF one-layer execution-plan gap receipt for the verified Qwen2.5 0.5B Q8_0 artifact, proving the planner routes layer-0 dense linear ops to dense_regular_llm_cuda while strict CUDA rejects unsupported non-linear layer ops with no CPU fallback, no dense GGUF inference, no Qwen token/decode/chat, no speedup, no full-residency, and no BitNet packed QK256 proof claim. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
