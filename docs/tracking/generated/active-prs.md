<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-POWER-005 | #5924 | `codex/lunar-lake/LNL258V-POWER-005-battery-telemetry-guard` | Add a fail-closed battery-mode telemetry guard for Lunar Lake low_power evidence so telemetry-context --require-battery records requirement status, writes a blocked receipt when the current machine is still on AC, fails strict mode after writing that receipt, and wires the blocked battery telemetry receipt into low_power power-profile/regression/comparison/audit artifacts while keeping low_power unpromoted and preserving no inference, speedup, power-advantage, route-promotion, native accelerator, acceleration, or BitNet QK256/I2_S behavior-change claims. |
