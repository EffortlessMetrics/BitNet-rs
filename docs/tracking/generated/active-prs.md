<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-067 | #109 | `codex/slm-cpu-067-exact-tensor-q8-runtime-hook` | Implement or produce a blocker for an exact-tensor packed Q8_0 runtime compute hook for `layers.0.attention.q_proj.weight` only, using the accepted SLM-CPU-065/066 sidecar evidence and preserving eager F32 Candle as the default path. Any opt-in runtime candidate must emit before/after strict CPU receipts proving unchanged model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook-selection identity, and fallback_used=false before it can be exported back to BitNet-rs release-surface review. The slice must not enable packed Q8_0 compute by default or claim speedup, sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
