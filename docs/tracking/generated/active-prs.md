<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-slm-performance | M4-SLM-PERF-006 | #4075 | `codex/apple-m4-slm-performance/M4-SLM-PERF-006-streaming-ux` | Add streaming token output, time-to-first-token receipts, quiet default logs, operator-friendly progress, and clear failure messages without changing backend claim boundaries. |
| intel-258v-platform | LNL258V-COMPARE-001 | #4076 | `codex/lunar-lake/LNL258V-COMPARE-001` | Document a Lunar Lake 258V same-machine comparison artifact that links separate platform, CPU, Arc 140V, and NPU receipts by artifact path, backend identity, runtime API, proof stage, fallback status, OS/power context, and missing-artifact state without merging CPU, GPU, or NPU proof claims. |
| nvidia-5070ti | CUDA-PROD-003 | #4073 | `codex/cuda-prod/CUDA-PROD-003-residency-coverage` | Add CUDA execution-residency coverage receipts for the strict RTX 5070 Ti answer path so QK256 linears, upload-once weights, KV cache, norms, RoPE, attention/softmax, LM head, sampling, host/device transfer accounting, and non-resident phases are visible without claiming speedup or full residency before coverage proves it. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
