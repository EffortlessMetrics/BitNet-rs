<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-082 | #361 | `codex/slm-cpu-082-repeated-packed-q8-receipts` | Capture or ingest the repeated i5-8250U Qwen3 Q8_0 warm-session receipt pack required by SLM-CPU-081 for the exact-tensor packed_q8_sidecar path. The slice must provide at least three baseline receipts and three candidate receipts for the same host, model SHA, GGUF tokenizer authority, prompt corpus, 4-thread operator profile, selected CPU backend/kernel identity, dense hook identity, and fallback_used=false; preserve prompt IDs, generated IDs, and decoded text; emit qwen3-slm-cpu-081-repeated-packed-q8-timing-classification.json with result improved, regressed, inconclusive, or not_claimed; and keep the default runtime unchanged unless a later promotion item accepts the repeated evidence. It must not claim sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
