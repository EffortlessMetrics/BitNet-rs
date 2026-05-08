<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| cpu-proof | CPU-ANSWER-003 | #3993 | `codex/cpu-answer-003-reference-comparator` | Strict BitNet CPU answer runs can be compared against a known-good reference artifact for the same model SHA, tokenizer, prompt bytes/template/BOS policy, prompt token IDs, generated token IDs, decoded text, and first-step top-k/logit evidence so the first divergence is attributed to prompt/tokenizer/template, shared decode math, logits/sampler, or backend-specific execution; scalar-vs-AVX2 parity remains preserved and non-answer-ready artifacts stay diagnostic-only. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
