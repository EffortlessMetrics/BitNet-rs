<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU258V-008 | #4008 | `codex/intel-258v-platform/CPU258V-008-answer-case-filter` | Add an answer-corpus case-id filter so 258V answer-template refreshes can run one bounded corpus case at a time, preserving full corpus identity plus selected case IDs in the aggregate receipt without answer-quality, parity, speed, Arc, or NPU claims. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
