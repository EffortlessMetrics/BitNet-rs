<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-028 | #5384 | `codex/slm-cpu-028-q4-planning-gates` | Define the bounded Kaby Lake Q4_K_M/Q4_K_S expansion plan after the Qwen3 Q8_0 proof-appliance profile and Qwen2.5 Q8_0 second-model sanity evidence. The slice must identify candidate GGUF artifacts only by pinned SHA256 and metadata, define the strict load/tokenizer/backend/fallback gates, require constrained corpus, multi-token determinism, warm-session receipts, operator-profile timing/memory/storage context, and before/after generated-ID preservation before any Q4 support claim. It must not implement Q4 runtime support or claim broad answer quality, sustained throughput, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
