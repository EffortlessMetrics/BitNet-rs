<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-OP-010 | #5953 | `codex/lunar-lake/LNL258V-OP-010-structured-blocked-ask` | Make fail-closed Lunar Lake auto-route blocked ask receipts expose structured route-selection evidence for no-promoted-route profiles, including candidate_routes, promotion_status, route_reason, why_not_cpu, why_not_gpu, and why_not_npu at top level and in route_selection; make strict regression require those structured fields; refresh the committed low_power blocked ask, regression-v2, and operator-comparison receipts while preserving no inference, no route promotion, no speedup, no power-advantage, no native accelerator, no acceleration, and no BitNet QK256/I2_S behavior-change claims. |
