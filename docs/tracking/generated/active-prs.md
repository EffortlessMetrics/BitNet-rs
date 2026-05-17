<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-BENCH-002 | #5261 | `codex/lunar-lake/LNL258V-BENCH-002-total-response` | Normalize the Lunar Lake route-profile and cold/warm benchmark total-response surface for the promoted dense Qwen CPU route by deriving total_response_ms from existing model-load, tokenizer-load, tokenize, prefill, and decode timing fields when no explicit latency total is present, then refresh the route-profile, benchmark, durability, regression-v2, and comparison artifacts without running inference, changing route promotion, claiming speedup/acceleration, promoting OpenVINO GPU/NPU, or changing BitNet QK256/I2_S behavior. |
