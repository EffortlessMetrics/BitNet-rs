<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-POWER-016 | #6251 | `codex/lunar-lake/LNL258V-POWER-016-low-power-thermal-preflight` | Clarify the low_power battery runbook after live POWER-013 AC/thermal preflights so operators check battery mode, strict telemetry-context --require-battery, MSAcpi thermal access, and Windows thermal-zone counter values before route samples. The runbook must state that access-denied MSAcpi and zero-valued thermal-zone counters are preserved as thermal-unavailable blockers, not measured-temperature evidence, while still keeping low_power unpromoted until benchmark-qualified battery-mode power advantage exists. Preserve no new battery evidence, no route promotion, no speedup claim, no power-advantage claim, no measured-temperature claim, no native accelerator proof, no broad quality claim, no Qwen3 Lunar Lake promotion, and no BitNet QK256/I2_S behavior-change claim. |
