<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-058 | #6070 | `codex/slm-cpu-058-dense-hook-before-after-receipts` | Capture or ingest the next Qwen3 Q8_0 dense-hook before/after warm-session receipt pack after SLM-CPU-057. The pack must compare the eager_f32_candle Qwen3 Q8_0 Kaby behavior oracle against the selected dense-hook path or explicitly record that packed Q8_0 sidecar compute remains disabled. It must prove identical model SHA, tokenizer source/strictness, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook-selection identity, and fallback_used=false before any packed sidecar compute can be enabled. It must preserve speedup_claim=false unless a separate bounded timing item records timing evidence, and must not claim sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
