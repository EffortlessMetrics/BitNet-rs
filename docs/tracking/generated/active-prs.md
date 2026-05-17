<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-CPU-SLM-PERF-002 | #5320 | `codex/lunar-lake/LNL258V-CPU-SLM-PERF-002-resident-cpu` | Add a no-new-inference Lunar Lake dense Qwen CPU resident-session artifact that reads the existing CPU phase-attribution and repeated warm-session receipts; records model/tokenizer loaded once, resident prompt-loop timing by regression_tiny/ask_short/ask_normal, prompt-loop fallback/answer/determinism status, cold one-off reference timing, and resident/no-reload gaps without changing route promotion or claiming speedup, power advantage, Arc/NPU acceleration, or BitNet QK256/I2_S behavior. |
