<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-078 | #277 | `codex/slm-cpu-078-packed-matvec-aligned-path` | Use the SLM-CPU-077 real post-bridge counter artifact to reduce or tightly classify the dominant packed Q8_0 exact-tensor matvec compute cost for `layers.0.attention.q_proj.weight`. The slice may optimize only the opt-in exact-tensor packed sidecar path and must keep eager_f32_candle as the default runtime. Any runtime change must preserve prompt IDs, generated IDs, decoded text, model SHA, tokenizer source/strictness, selected CPU backend/kernel identity, dense hook identity, and fallback_used=false against the Qwen3 Q8_0 appliance oracle before claiming even a bounded improvement. If no safe optimization lands, emit a concrete blocker or next-target artifact. It must not enable packed Q8_0 by default, broaden beyond the exact tensor, start Q4/Q5 runtime support, claim sustained throughput or broad answer quality, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths. |
