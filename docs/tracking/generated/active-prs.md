<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| model-artifacts | MODEL-CAPS-001 | #4205 | `codex/model-caps-001-dense-verify` | Expose dense Qwen SLM capability summaries through `bitnet model verify` and cache/fetch metadata so Qwen Q8_0 and Q4_K_M artifacts have explicit model family, artifact class, tokenizer/prompt authority, route boundary, permitted claims, and required receipts without changing runtime inference, tokenizer, loader, transformer, QK256, CUDA, dense GGUF, server, residency, or speed-claim behavior. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
