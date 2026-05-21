<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-069 | #124 | `codex/slm-cpu-069-packed-q8-locality-root-cause` | Use the SLM-CPU-068 timing regression evidence to localize why the exact-tensor packed Q8_0 sidecar path for `layers.0.attention.q_proj.weight` is slower than eager F32 Candle on the bounded Qwen3 Q8_0 4-thread i5-8250U artifact. The slice must preserve eager_f32_candle as the default runtime, keep packed sidecar execution opt-in and exact-tensor scoped, identify whether the next fix belongs in packed-block decode, matvec locality, scratch allocation, selector overhead, or receipt/timing instrumentation, and emit a machine-checkable root-cause or next-target artifact. It must not change generated IDs/text, weaken strict tokenizer/backend/fallback receipts, claim speedup, enable packed Q8_0 by default, start Q4/Q5 runtime support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths. |
