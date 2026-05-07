<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-local-answer | M4-QA-001 | #3904 | `codex/apple-m4-local-answer/M4-QA-001-output-smoke` | Add a multi-prompt Apple M4 CPU/NEON local-answer smoke suite that runs real GGUF and tokenizer paths under strict mode, requires generated_tokens >= 16, valid UTF-8, non-empty output, non-degenerate token variation, and a receipt with explicit fallback status. |
| apple-m4-local-answer | M4-QA-ROOT-001 | #3908 | `codex/apple-m4-local-answer/M4-QA-ROOT-001-bitnetcpp-parity` | Compare the same real GGUF, tokenizer, prompt template, prompt, and greedy settings against bitnet.cpp/reference behavior; either produce token/logit parity evidence for the first divergence and fix the Rust path, or prove the local GGUF artifact itself also garbles under the reference implementation. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
