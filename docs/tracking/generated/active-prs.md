<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-CPU-SLM-PERF-001 | #5315 | `codex/lunar-lake/LNL258V-CPU-SLM-PERF-001-phase-attribution` | Add a no-new-inference Lunar Lake dense Qwen CPU phase-attribution artifact that reads the existing CPU warm-session, cold/warm benchmark, and dense SLM OpenVINO phase comparison receipts; classifies cold one-off model-load/tokenize/prefill/first-token/decode shares, indexes warm prefill_512 and decode_128 rates, records OpenVINO CPU timing context only, recommends resident CPU/OpenVINO CPU follow-ups, and preserves no route-promotion, no speedup, no power-advantage, no Arc/NPU acceleration, and no BitNet QK256/I2_S behavior-change boundaries. |
