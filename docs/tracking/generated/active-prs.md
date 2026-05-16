<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-QUAL-005 | #5124 | `codex/lunar-lake/LNL258V-QUAL-005-qwen-gate-normalization` | Normalize the bounded Lunar Lake dense Qwen answer-corpus quality path for the leading assistant-role punctuation artifact already handled by warm-session scoring, rerun the CPU corpus-v2 receipt, refresh the route-profile/regression/diagnosis artifacts, and preserve the no-broad-quality, no-speedup, no-route-promotion, no-Arc/NPU-execution, and no BitNet QK256/I2_S behavior-change boundary. |
| slm-cpu | SLM-CPU-022 | #5140 | `model/slm-cpu-022-smollm2-reference-comparator` | Define the SmolLM2 360M reference-compatible first-token/top-k comparator contract after the SLM-CPU-021 wrong-first-token diagnosis, and add CLI fixture coverage showing `reference-compare` validates the exact SmolLM2 model-family/top-k artifact shape and fails closed under `--require-match` when it diverges. This support slice must not claim a fresh external reference run, CPU answer readiness, CUDA planning, CUDA execution, speedup, server readiness, broad dense GGUF support, or BitNet QK256 behavior. |
