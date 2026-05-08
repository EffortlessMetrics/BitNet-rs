<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-productization | M4-PROD-001 | #4017 | `codex/apple-m4-productization/M4-PROD-001-user-facing-baseline` | Document the working Rust-native Apple M4 CPU/NEON SLM local-answer baseline, including the current warm-session command, expected model artifact, receipt fields, failure boundaries, and unsupported claims. |
| cpu-proof | CPU-ANSWER-006 | #4005 | `codex/cpu-answer-006-reference-token-artifact` | A Microsoft BitNet.cpp reference-divergence artifact records the MODEL-ARTIFACT-007 prompt envelope, BOS policy, external Llama-BPE tokenizer/pre-tokenizer authority, prompt token IDs, generated token IDs, decoded text, and first-step top-k/logit evidence where available, so strict Rust CPU failures can be classified as prompt/tokenizer divergence, shared decode/logits divergence, or backend-specific execution without claiming answer quality. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
