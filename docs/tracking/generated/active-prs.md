<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| cpu-proof | CPU-ANSWER-002 | #3906 | `codex/cpu-answer-002-full-decode-parity` | Strict CPU answer runs can compare scalar and AVX2 full-decode outputs for the same real GGUF, tokenizer, prompt, greedy settings, prompt token IDs, generated token IDs, decoded text, and per-step logits/top-k evidence so AVX2 divergence is separated from shared decode correctness; the 258V CPU is the lead machine for new BitNet CPU answer parity artifacts. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
