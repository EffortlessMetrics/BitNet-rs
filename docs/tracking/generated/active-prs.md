<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-PROD-004 | #4093 | `codex/cuda-prod/CUDA-PROD-004-benchmark-baseline` | Add strict RTX 5070 Ti answer-path benchmark receipts after residency coverage, measuring same-model CPU AVX-512 and CUDA runs across deterministic profiles with model load, tokenizer load, CUDA context init, weight upload, prompt render/tokenize, prefill, first token, steady decode, kernel time, host/device transfer accounting, VRAM, power/temperature where available, and speedup_claim=false until the benchmark evidence is explicitly accepted. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
