<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-REG-001 | #4624 | `codex/lunar-lake/LNL258V-REG-001-local-regression` | Add a Lunar Lake local regression bundle command that reads the committed operator-readiness receipt, verifies dense Qwen CPU default routing, BitNet CPU reference routing, bounded OpenVINO GPU/NPU candidate routing, Arc/NPU claim boundaries, no hidden fallback, and no acceleration claim, then commits a 258V regression bundle artifact without running inference or claiming speedup, broad answer quality, Arc/NPU acceleration, full BitNet inference on accelerators, or QK256 accelerator decode. |
| nvidia-5070ti | CUDA-PROD-010 | #4616 | `cuda/bitnet-i2s-benchmark-qualification` | Add a governed BitNet I2_S/QK256 CUDA product benchmark qualification receipt and report for the five target user profiles, wire report-only bench output for text/json/csv against that receipt, preserve receipts explain coverage, and keep every profile speedup-unqualified until fresh profile-specific benchmark evidence exists. |
