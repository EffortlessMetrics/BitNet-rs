<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-000 | #3902 | `codex/slm-cpu-000-8250u-lane` | Define a separate 8250U dense SLM CPU proof lane with explicit target policy, dense GGUF requirements, tokenizer authority rules, architecture adapter boundaries, receipt fields, and claim boundaries. The first candidate policy prefers Qwen2.5-0.5B-Instruct GGUF Q4_K_M or Q8_0 after verifying exact artifact metadata. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
