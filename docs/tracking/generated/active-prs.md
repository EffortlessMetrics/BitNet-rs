<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| cpu-proof | CPU-ANSWER-004 | #3995 | `codex/cpu-answer-004-bitnetcpp-template` | Strict Rust CPU answer-corpus runs use the MODEL-ARTIFACT-007 answer-ready authority: the Microsoft BitNet.cpp reference prompt envelope `User: <question><|eot_id|>Assistant:`, external tokenizer/pretokenizer provenance, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel, fallback=false, and reference-divergence evidence before any BitNet answer-quality claim. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
