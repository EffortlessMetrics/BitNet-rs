<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-073 | #195 | `codex/slm-cpu-073-packed-q8-regression-root-cause` | Use the SLM-CPU-072 real six-prompt i5-8250U timing evidence to localize why the opt-in exact-tensor `layers.0.attention.q_proj.weight` packed_q8_sidecar path remains slower than the default eager_f32_candle path after the SLM-CPU-070 block-local matvec prototype. The slice must preserve eager_f32_candle as the default runtime, keep packed sidecar execution opt-in and exact-tensor scoped, identify whether the next fix belongs in payload decode, packed-block matvec locality, scratch allocation, selector/dispatch overhead, tensor layout, or timing instrumentation, and emit a machine-checkable root-cause or next-target artifact. It must not change generated IDs/text, weaken strict tokenizer/backend/fallback receipts, claim speedup, enable packed Q8_0 by default, start Q4/Q5 runtime support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths. |
