<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-057 | #6045 | `codex/slm-cpu-057-dense-hook-receipt-gate` | Use the SLM-CPU-056 production dense-linear hook boundary to add or prove the next receipt gate for Qwen3 Q8_0 dense hook selection before any packed Q8_0 sidecar compute can be enabled. The slice must keep eager_f32_candle as the default behavior oracle unless before/after Qwen3 Q8_0 warm-session receipts prove identical model SHA, tokenizer source/strictness, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook selection identity, and fallback_used=false. If packed runtime compute remains disabled, emit a machine-checkable blocker artifact naming the remaining compute-kernel or receipt gap. It must not claim speedup, sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
