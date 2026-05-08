<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-PROD-002 | #4059 | `codex/cuda-prod/CUDA-PROD-002-warm-session-receipts` | Add a strict CUDA warm-session ask/chat path that loads the model once, initializes the RTX 5070 Ti CUDA context once, uploads BitNet weights once, serves multiple deterministic turns, and emits per-turn or session-summary receipts without broad chat, speed, server, or full-residency claims. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
