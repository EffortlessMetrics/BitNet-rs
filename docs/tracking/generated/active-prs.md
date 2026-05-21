<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-070 | #129 | `codex/slm-cpu-070-packed-q8-block-local-matvec` | Use the SLM-CPU-069 root-cause artifact to prototype or explicitly block a block-local packed Q8_0 matvec path for the exact `layers.0.attention.q_proj.weight` sidecar. The slice must preserve eager_f32_candle as the default runtime, keep packed sidecar execution opt-in and exact-tensor scoped, decode each Q8_0 block scale once per block rather than per weight where a runtime prototype is attempted, and compare against the existing eager F32 behavior oracle with identical prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook-selection identity, and fallback_used=false before any timing interpretation. It must not claim speedup unless a separate bounded timing artifact supports it, enable packed Q8_0 by default, broaden beyond the exact tensor, start Q4/Q5 runtime support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths. |
