<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-060 | #6085 | `codex/slm-cpu-060-payload-bearing-q8-sidecar-hook` | Add the first evidence-scoped payload-bearing packed Q8_0 sidecar hook contract for one Qwen3 Q8_0 dense-linear tensor path, or emit an implementation blocker if the transformer/model API cannot safely carry payload bytes to the dense-linear callsite. Any runtime candidate must remain opt-in/disabled by default, preserve the eager_f32_candle behavior oracle, and require before/after warm-session receipts proving identical model SHA, tokenizer source/strictness, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook-selection identity, and fallback_used=false before broader selection. The item must not claim speedup, sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
