<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-UX-001 | #4185 | `codex/cuda-ux-receipts-explain` | Add a user-facing `bitnet receipts explain` command that reads existing BitNet and dense CUDA JSON receipts, supports `--latest` under target/bitnet/receipts, and prints or emits a compact proof summary covering artifact kind, claim, model, backend, execution plan route, kernels, quality, timing, residency, and claim limits without changing inference, tokenizer, loader, kernel, transformer, benchmark, or server behavior. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
