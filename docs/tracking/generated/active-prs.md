<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-001 | #3905 | `codex/slm-cpu-001-model-manifest` | Add a model candidate manifest, artifact policy, and 8250U runbook. The first target is Qwen2.5-0.5B-Instruct GGUF with Q4_K_M preferred and Q8_0 optional, but only after exact artifact path, SHA256, GGUF architecture string, tokenizer metadata, tensor naming, and chat template policy are verified. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
