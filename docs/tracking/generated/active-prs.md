<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-080 | #332 | `codex/slm-cpu-080-kaby-performance-dashboard-refresh` | Refresh the Kaby Lake Qwen3 Q8_0 CPU performance dashboard after the merged prompt-token cache, KV-cache reuse, prefill-attribution, and post-aligned packed-matvec evidence. The dashboard must preserve the Qwen3 Q8_0 behavior oracle, record that the current operator default is evidence-scoped to 4 threads, identify the current allocation and packed-matvec next targets, and keep speedup, sustained-throughput, Q4/Q5 runtime, server, accelerator, Qwen3.5, and BitNet QK256 claims false. |
