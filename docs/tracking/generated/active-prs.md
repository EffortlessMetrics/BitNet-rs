<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-GOAL-AUDIT-040 | #6259 | `codex/lunar-lake/LNL258V-GOAL-AUDIT-040-power006-wording-guard` | Add a campaign check guard that rejects stale no-inference objective-artifact wording which makes POWER-006 the current low_power blocker after POWER-013 has become the active blocker. Scope the guard to the intel-258v-platform audit/checklist artifacts, keep POWER-006 historical evidence wording allowed, and preserve no inference, model load, fallback behavior change, route promotion, speedup claim, power-advantage claim, measured-temperature claim, native accelerator proof, broad quality claim, Qwen3 Lunar Lake promotion, or BitNet QK256/I2_S behavior-change claim. |
