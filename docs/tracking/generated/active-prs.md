<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-014 | #4216 | `codex/cuda-dense-014-gap-audit` | Extend the dense GGUF one-layer planner receipt with a gap_audit section that records unsupported strict CUDA non-linear ops, dependency notes, candidate order, and not-executed residency/timing status while keeping dense GGUF inference, Qwen token/decode/chat, speedup, full-residency, BitNet packed proof, tokenizer, loader, transformer, kernel, and server claims false. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
