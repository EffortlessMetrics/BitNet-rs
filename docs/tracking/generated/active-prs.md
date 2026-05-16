<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-017 | #5041 | `codex/slm-cpu-017-second-model-sanity` | Add a bounded second small dense GGUF model sanity check after the Qwen3-0.6B Q8_0 appliance profile. The slice must select a non-Qwen3 candidate only by verified artifact metadata, pinned SHA256, GGUF architecture, tokenizer authority, quant format, and tensor naming; run the strict CPU preflight and the smallest diagnosable strict CPU answer or failure receipt the current command surface supports; record prompt IDs, generated IDs when generated, decoded text when generated, selected backend/kernel, loader mode, tokenizer source/strictness, and fallback_used=false. It must preserve the Qwen3 Q8_0 appliance profile as the baseline and must not claim broad answer quality, sustained throughput, Q4/Q5 expansion, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
