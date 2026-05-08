<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| model-artifacts | MODEL-ARTIFACT-005 | #3977 | `codex/model-artifacts/MODEL-ARTIFACT-005-authority-dimensions` | Split artifact authority into explicit target alignment, runner authority, tokenizer/pre-tokenizer authority, prompt-suite result, and per-lane unblock fields so alternate-quant control evidence cannot be confused with the official Microsoft I2_S CUDA target. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
