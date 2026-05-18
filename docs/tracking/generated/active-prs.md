<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-030 | #5457 | `codex/slm-cpu-030-prompt-setup-attribution` | Add bounded warm-session allocation attribution for the Qwen3 Q8_0 Kaby prompt setup boundary after SLM-CPU-029. The slice must keep the existing prompt_setup total for receipt continuity while breaking it down into buffer reset, token seeding, KV-cache creation, and sampler setup subcomponents so the next optimization target is evidence-scoped. It must preserve generated IDs, decoded text, strict GGUF tokenizer authority, selected CPU backend/kernel, model SHA, and fallback=false when real artifacts are regenerated, and must not claim sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
| apple-m4-inference-excellence | M4-OPS-UX-002 | #5478 | `codex/apple-m4-inference-excellence/M4-OPS-UX-002-explain-open` | Add explanation/open affordances for report-refresh and regression-dashboard outputs so operators can see why a group is comparable, warning, failed, or insufficient_history. |
