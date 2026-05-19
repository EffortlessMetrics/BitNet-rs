<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-REG-011 | #6027 | `codex/lunar-lake/LNL258V-REG-011-route-model-identity-gate` | Make strict Lunar Lake regression and operator readiness fail closed when route-profile or cold/warm benchmark evidence loses route/model identity, tokenizer/template coverage, or model-hash-or-explicit-gap coverage; preserve the OpenVINO IR no-local-binary-SHA gap as explicit evidence rather than a fake hash; refresh route-profile, benchmark, power-profile, readiness, regression-v2, comparison, and generated tracking artifacts; preserve no new inference, no route promotion, no speedup claim, no power-advantage claim, no measured-temperature claim, no native accelerator claim, no broad quality claim, and no BitNet QK256/I2_S behavior change. |
