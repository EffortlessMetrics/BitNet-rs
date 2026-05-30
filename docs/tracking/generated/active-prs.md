<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-POWER-017 | #6253 | `codex/lunar-lake/LNL258V-POWER-017-low-power-plan-thermal-preflight` | Update the machine-readable Lunar Lake low_power battery plan so the generated plan mirrors the POWER-016 runbook thermal preflight: include MSAcpi and Windows thermal-zone counter commands, treat MSAcpi access-denied and CookedValue=0 as thermal-unavailable blockers rather than measured-temperature evidence, add an explicit measured_temperature_claim=false boundary, refresh the committed low-power plan JSON, and preserve no new inference, no route promotion, no speedup claim, no power-advantage claim, no measured-temperature claim, no native accelerator proof, no broad quality claim, no Qwen3 Lunar Lake promotion, and no BitNet QK256/I2_S behavior-change claim. |
