<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-CPU-SLM-PERF-003 | #5324 | `codex/lunar-lake/LNL258V-CPU-SLM-PERF-003-openvino-cpu-compare` | Add a no-new-inference Lunar Lake dense Qwen CPU runtime-comparison artifact that reads the resident Rust GGUF CPU receipt plus OpenVINO CPU corpus-v2 and phase-runner receipts; compares fallback, answer-gate/profile quality, load/construct timing, generation/TTFT/tokenization timing context, and profile blockers for Rust GGUF CPU versus OpenVINO CPU; keeps Rust GGUF CPU as default when OpenVINO CPU profile quality fails or phase coverage is not benchmark-qualified; and preserves no route-promotion, no speedup, no power-advantage, no Arc/NPU acceleration, and no BitNet QK256/I2_S behavior-change boundaries. |
