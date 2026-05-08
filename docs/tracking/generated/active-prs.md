<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| cpu-proof | CPU-ANSWER-005 | #4003 | `codex/cpu-answer-005-tokenizer-authority` | Strict Rust CPU answer receipts record the MODEL-ARTIFACT-007 external Llama-BPE tokenizer/pre-tokenizer authority instead of `unknown` when an explicit or sibling Llama-3 tokenizer is used; failed answer-corpus artifacts remain diagnostic and no answer-quality, throughput, server, GPU, or NPU claim is made. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
