<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-COMPARE-002 | #4110 | `codex/lunar-lake/LNL258V-COMPARE-002` | Refresh the Lunar Lake same-machine comparison index after the post-mechanics CPU reference bundle and the next Arc 140V native OpenCL parity receipt, preserving independent lane claims, missing-artifact states, fallback status, and no platform performance or acceleration claims. |
| nvidia-5070ti | CUDA-DENSE-003 | #4119 | `codex/cuda-dense-003-tensor-residency` | Add fixture-level dense regular-LLM CUDA tensor residency evidence for the RTX 5070 Ti FP16 GEMM path, proving input/output tensors are CUDA device buffers for the launch, transfer byte accounting matches kernel stats, fallback_used=false, and dense receipts still cannot satisfy BitNet packed I2S/QK256, speedup, dense GGUF inference, persistent session residency, or full CUDA residency claims. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
