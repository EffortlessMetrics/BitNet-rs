<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-slm-performance | M4-SLM-PERF-006 | #4075 | `codex/apple-m4-slm-performance/M4-SLM-PERF-006-streaming-ux` | Add streaming token output, time-to-first-token receipts, quiet default logs, operator-friendly progress, and clear failure messages without changing backend claim boundaries. |
| nvidia-5070ti | CUDA-PROD-003 | #4073 | `codex/cuda-prod/CUDA-PROD-003-residency-coverage` | Add CUDA execution-residency coverage receipts for the strict RTX 5070 Ti answer path so QK256 linears, upload-once weights, KV cache, norms, RoPE, attention/softmax, LM head, sampling, host/device transfer accounting, and non-resident phases are visible without claiming speedup or full residency before coverage proves it. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
