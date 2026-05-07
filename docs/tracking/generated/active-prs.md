<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-BITNET-009 | #3837 | `codex/cuda-bitnet-009-upload-once-routed-proof` | Route the strict RTX 5070 Ti CUDA inference proof through persistent CUDA BitNet weight handles so receipts prove weights_uploaded_once=true, per_token_weight_upload=false, qk256_gemv_cuda invocations greater than zero, and zero CPU fallback. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
