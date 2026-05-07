<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-slm-answer | SLM-M4-002 | #3930 | `codex/apple-m4-slm-answer/SLM-M4-002-validate-artifact` | Validate a sub-1 GiB dense instruct GGUF under a reference runner against the M4 SLM prompt suite, record source, SHA256, size, GGUF architecture, quantization, tokenizer metadata, pre-tokenizer authority, prompt template, and reference output, and reject candidates that fail quality. |
| slm-cpu | SLM-CPU-002B | #3926 | `codex/slm-cpu-002b-dense-q8-gguf` | Add dense standard GGUF Q8_0/Q*_K adapter or dequantization support sufficient for the verified Qwen3-0.6B Q8_0 artifact to pass strict load without compatibility fallback, while preserving explicit failure for unsupported dense quantization variants. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
