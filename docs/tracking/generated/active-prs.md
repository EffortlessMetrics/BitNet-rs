<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU258V-004 | #3981 | `codex/intel-258v-platform/CPU258V-004-phase-thresholds` | Require 258V CPU phase benchmark receipts to keep decode_128 and prefill_512 as not_run unless supplied strict CPU proof receipts meet the corresponding generated-token and prompt-token thresholds, preserving explicit gaps rather than promoting proxy evidence. |
| model-artifacts | MODEL-ARTIFACT-007 | #3988 | `codex/model-artifacts/MODEL-ARTIFACT-007-msft-bitnetcpp-external-pretokenizer` | Record Microsoft BitNet.cpp reference-runner evidence showing the official Microsoft I2_S GGUF passes the committed deterministic answer corpus when the externally supplied Microsoft tokenizer pre-tokenizer authority is injected with tokenizer.ggml.pre=llama-bpe, without changing Rust runtime behavior or making backend answer claims. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
