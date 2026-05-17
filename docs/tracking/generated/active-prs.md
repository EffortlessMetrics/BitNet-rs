<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-025 | #5357 | `codex/slm-cpu-025-logits-extraction-reuse` | Remove or explicitly isolate the remaining Qwen3-0.6B Q8_0 warm-session logits extraction allocation after SLM-CPU-024 without changing generated IDs or strict receipt provenance. The slice must preserve the Qwen3 Q8_0 4-thread behavior oracle, keep model SHA, prompt IDs, generated IDs, decoded text, strict GGUF tokenizer authority, selected CPU backend, and fallback=false identical to the recorded baseline, and emit before/after allocation evidence or explicitly documented unavailable counters. It must not claim sustained throughput, broad answer quality, Q4/Q5 quant expansion, a second dense model, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
