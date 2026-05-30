<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-POWER-015 | #6241 | `codex/lunar-lake/LNL258V-POWER-015-comparative-energy-proxy` | Add explicit comparative route coverage to the Lunar Lake low_power energy-proxy path so a single battery before/after matrix receipt can name the CPU, OpenVINO GPU, and OpenVINO NPU candidate routes it covers. Keep old single-route receipts classified as recorded_not_comparative, make power-profile accept accepted_comparative_proxy only when covered routes include every candidate and fallback-free CPU/GPU/NPU low_power ask receipts are indexed, refresh the operator battery plan/runbook command, and preserve no new inference, no low_power route promotion, no speedup, no power-advantage, no native accelerator, no measured-temperature, and no BitNet QK256/I2_S behavior-change claim. |
