<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| model-artifacts | BITNET-CONTRACT-003 | #4201 | `codex/bitnet-contract-003-verify-matrix` | Expose the full BitNet-family model contract matrix through `bitnet model contracts` and make `bitnet model verify` fail closed with a contract summary for known BitNet contracts that do not have supported artifact identity and SHA256 metadata, without changing runtime inference, tokenizer, loader, transformer, QK256, CUDA, dense GGUF, server, or speed-claim behavior. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
