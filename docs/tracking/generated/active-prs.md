<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-inference-excellence | M4-SERVE-EX-002 | #112 | `codex/apple-m4-inference-excellence/M4-SERVE-EX-002-streaming-failure-semantics` | Prove local-server streaming and failure semantics across dense SLM and enabled BitNet serve paths: partial token streaming, client cancellation, timeout stage, invalid request, missing cache, per-request receipt export, and no-response failure receipts. |
| slm-cpu | SLM-CPU-068 | #120 | `codex/slm-cpu-068-exact-hook-timing-classification` | Classify the SLM-CPU-067 exact-tensor packed Q8_0 runtime hook timing evidence without broadening the runtime claim. The slice must compare the committed before/after Qwen3 Q8_0 warm-session receipts for `layers.0.attention.q_proj.weight`, record that generated IDs, decoded text, model SHA, strict GGUF tokenizer authority, selected CPU backend, dense hook-selection identity, and fallback_used=false are preserved, and state whether the opt-in packed sidecar path improves, regresses, or is inconclusive for the bounded two-prompt 4-thread artifact. It must keep eager F32 Candle as the default path and must not claim speedup unless the measured artifact supports it under the existing claim boundary. |
