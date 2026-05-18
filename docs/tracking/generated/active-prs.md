<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-032 | #5514 | `codex/slm-cpu-032-prompt-token-cache` | Reuse rendered prompt token IDs across repeated Qwen3 Q8_0 Kaby warm-session prompts so repeated corpus cases avoid redundant tokenizer.encode work while preserving generated IDs, decoded text, strict GGUF tokenizer authority, selected CPU backend/kernel, model SHA, and fallback=false when real artifacts are regenerated. Receipts must record prompt token cache policy, hit/miss counts, and cache entry count, and must not claim sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
