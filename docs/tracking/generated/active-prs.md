<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-PLANNER-001 | #4127 | `codex/cuda-planner-001-model-aware` | Add a model-aware dispatch planner contract that keeps BitNet QK256 CUDA and dense regular-LLM CUDA routes separate, rejects unsupported strict CUDA fallback explicitly, and does not change kernels, model math, tokenizer, loader, transformer, or server behavior. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
