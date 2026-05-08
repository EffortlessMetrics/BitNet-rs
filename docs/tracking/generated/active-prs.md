<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-PROD-001 | #4054 | `codex/cuda-prod/CUDA-PROD-001-strict-ask-receipt-default` | Make strict `bitnet ask` validate backend, fallback, and answer quality even when the user does not pass --receipt-out by writing a default answer receipt for strict CPU/CUDA ask runs, preserving no speed claim and no broad chat claim. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
