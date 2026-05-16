<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-QUAL-004 | #5100 | `codex/lunar-lake/LNL258V-QUAL-004-qwen-corpus-v2-diagnosis` | Add a diagnostic Lunar Lake dense Qwen CPU corpus-v2 failure artifact that reads the committed CPU corpus-v2 execution receipt plus route-profile comparison, classifies failed cases by profile/category/failure class, records route blockers and recommended next fixes, and preserves the no-new-inference, no-broad-quality, no-speedup, no-route-promotion, no-Arc/NPU-execution, and no BitNet QK256/I2_S behavior-change boundary. |
| slm-cpu | SLM-CPU-020 | #5108 | `model/slm-cpu-020-smollm2-cpu-sanity` | Retry strict SmolLM2 360M Q8_0 CPU sanity after exact metadata-scoped normalization validation. The slice must keep the generic llama normalization guard fail-closed, preserve exact SmolLM2 artifact and metadata scope, record tokenizer/prompt/generation reachability or the next blocker with fallback_used=false, and keep CPU answer readiness false unless the bounded quality gate passes. |
