<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-055 | #6008 | `codex/slm-cpu-055-production-dense-linear-hook-contract` | Add the first production dense-linear hook contract after SLM-CPU-054 so transformer dense linear calls can receive an explicit eager-F32 selection or a selected Q8_0 sidecar descriptor without enabling packed sidecar compute by default. The slice must preserve eager_f32_candle as the default Qwen3 Q8_0 Kaby behavior oracle unless before/after receipts prove identical model SHA, tokenizer source/strictness, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, and fallback_used=false. If packed runtime compute remains disabled, it must emit a machine-checkable hook-contract or blocker artifact naming the remaining dispatch/receipt gap. It must not claim speedup, sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
