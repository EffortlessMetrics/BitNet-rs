<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-inference-excellence | M4-BENCH-008 | #5902 | `codex/apple-m4-inference-excellence/M4-BENCH-008-repeat-harness` | Implement the repeat-run M4 benchmark variance harness required before M4-BENCH-005 can publish live variance evidence: expose `bitnet mac benchmark --repeat <n>` or a documented equivalent, write a benchmark-variance receipt with run count, sample count, variance band, outlier handling, threshold derivation, invalid-comparison reasons, model/profile identity, backend/fallback state, and receipt validation/tests. |
| intel-258v-platform | LNL258V-PROFILE-RUN-004 | #5898 | `codex/lunar-lake/LNL258V-PROFILE-RUN-004-cpu-heavy-profile-run` | Run explicit dense Qwen Rust GGUF CPU prefill_heavy and decode_heavy profile cases on the 258V, record same-machine CPU baseline timing with prompt/output token counts that satisfy the route-profile thresholds, feed that timing into route-profile/benchmark/regression/readiness/comparison/excellence artifacts, and keep route promotion, speedup, power advantage, OpenVINO GPU/NPU promotion, native accelerator, and BitNet QK256/I2_S claims unchanged. |
