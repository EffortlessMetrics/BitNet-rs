<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-NPU-COLD-001 | #5332 | `codex/lunar-lake/LNL258V-NPU-COLD-001-cold-start-diagnosis` | Add a no-new-inference Lunar Lake OpenVINO NPU cold-start diagnosis artifact that reads existing NPU operator-ask, OpenVINO phase-runner, phase-comparison, and corpus-v2 receipts; separates pipeline load/device compile timing from hot generation, first-token, throughput, and corpus-v2 blocker evidence; classifies whether the NPU cold path is load dominated; recommends cache/resident/power follow-up work; and preserves no route-promotion, no speedup, no power-advantage, no acceleration, no native NPU inference beyond OpenVINO GenAI, and no BitNet QK256/I2_S behavior-change boundaries. |
