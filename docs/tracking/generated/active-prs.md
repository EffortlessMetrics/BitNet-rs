<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| model-artifacts | BITNET-CONTRACT-004 | #4203 | `codex/bitnet-contract-004-architecture-summary` | Expose architecture support rows in the BitNet-family contract summaries emitted by `bitnet model contracts` and contract-aware `bitnet model verify`, so x86, ARM, supported-reference, proof-required, listed-verify-runner, and upstream-unsupported routes are visible without changing runtime inference, tokenizer, loader, transformer, QK256, CUDA, dense GGUF, server, or speed-claim behavior. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
