<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-COMPARE-002 | #4110 | `codex/lunar-lake/LNL258V-COMPARE-002` | Refresh the Lunar Lake same-machine comparison index after the post-mechanics CPU reference bundle and the next Arc 140V native OpenCL parity receipt, preserving independent lane claims, missing-artifact states, fallback status, and no platform performance or acceleration claims. |
| nvidia-5070ti | CUDA-BITNET-PERF-002 | #4112 | `codex/cuda-bitnet-perf-002-repeated-strict-ask` | Add repeated strict ask benchmark receipts for the official Microsoft I2_S model, comparing same-model 9950X3D CPU AVX-512 and RTX 5070 Ti CUDA runs with deterministic policy, fallback_used=false, measured QK256 timing/transfer counters, and speedup_claim=false pending explicit benchmark review. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
